#include "layers.h"
#include "layers_internal.h"

void _layer_conv_2d_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    const ts_layer_conv_2d_desc* cdesc = &desc->conv_2d;
    _layer_conv_2d_backend* conv = &out->conv_2d_backend;

    conv->kernel_size = cdesc->kernel_size;
    conv->stride_x = cdesc->stride_x;
    conv->stride_y = cdesc->stride_y;

    conv->input_shape = prev_shape;
    conv->padded_shape = prev_shape;

    if (cdesc->padding) {
        // Padding so that out_shape == in_shape when stride is 1
        conv->padded_shape.width += conv->kernel_size.width - 1;
        conv->padded_shape.height += conv->kernel_size.height - 1;
    }

    out->shape = ts_tensor_conv_shape(conv->padded_shape, conv->kernel_size, conv->stride_x, conv->stride_y);
    out->shape.depth = cdesc->num_filters;

    // Have to collapse one dimension because ts_tensors are only 3d
    ts_tensor_shape kernels_shape = {
        .width = conv->kernel_size.width * conv->kernel_size.height,
        .height = prev_shape.depth,
        .depth = cdesc->num_filters
    };

    conv->kernels = ts_tensor_create(arena, kernels_shape);
    conv->biases = ts_tensor_create(arena, out->shape);

    ts_u64 in_size = (ts_u64)prev_shape.width * prev_shape.height * prev_shape.depth;
    ts_u64 out_size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;
    ts_param_init(conv->kernels, cdesc->kernels_init, in_size, out_size);
    ts_param_init(conv->biases, cdesc->biases_init, in_size, out_size);

    if (desc->training_mode) { 
        ts_param_change_create(arena, &conv->kernels_change, kernels_shape);
        ts_param_change_create(arena, &conv->biases_change, out->shape);
    }
}
void _layer_conv_2d_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = { 0 };
    mg_arena* input_arena = NULL;
    if (cache != NULL) {
        scratch = mga_scratch_get(&cache->arena, 1);
        input_arena = cache->arena;
    } else {
        scratch = mga_scratch_get(NULL, 0);
        input_arena = scratch.arena;
    }

    ts_tensor* input = NULL;
    if (ts_tensor_shape_eq(in_out->shape, conv->padded_shape)){
        input = ts_tensor_copy(input_arena, in_out, false);
    } else {
        // Create padded shape
        input = ts_tensor_create(input_arena, conv->padded_shape);

        ts_u32 x_off = (conv->padded_shape.width - in_out->shape.width) / 2;
        ts_u32 y_off = (conv->padded_shape.height - in_out->shape.height) / 2;

        for (ts_u32 z = 0; z < in_out->shape.depth; z++) {
            for (ts_u32 y = 0; y < in_out->shape.height; y++) {
                for (ts_u32 x = 0; x < in_out->shape.width; x++) {
                    ts_u64 in_out_index = (ts_u64)x +
                        (ts_u64)y * in_out->shape.width +
                        (ts_u64)z * in_out->shape.width * in_out->shape.height;
                    ts_u64 input_index = (ts_u64)x + x_off +
                        (ts_u64)(y + y_off) * conv->padded_shape.width +
                        (ts_u64)z * conv->padded_shape.width * conv->padded_shape.height;

                    input->data[input_index] = in_out->data[in_out_index];
                }
            }
        }
    }

    if (cache != NULL) {
        ts_layers_cache_push(cache, input);
    }

    // Renaming for clarity
    ts_tensor* output = in_out;

    output->shape = l->shape;
    ts_tensor_fill(output, 0.0f);

    ts_tensor input_view = { 0 };
    ts_tensor output_view = { 0 };

    // Stores individual kernel of each iteration
    ts_tensor kernel_view = { .shape = conv->kernel_size };
    ts_tensor_shape kernels_shape = conv->kernels->shape;

    // Used for storing a conv output before adding to output
    ts_tensor* out_temp = ts_tensor_create(scratch.arena, (ts_tensor_shape){ output->shape.width, output->shape.height, 1 });

    for (ts_u32 o_z = 0; o_z < output->shape.depth; o_z++) {
        ts_tensor_2d_view(&output_view, output, o_z);

        for (ts_u32 i_z = 0; i_z < input->shape.depth; i_z++) {
            ts_tensor_2d_view(&input_view, input, i_z);

            ts_u64 kernel_index = (ts_u64)i_z * kernels_shape.width + (ts_u64)o_z * kernels_shape.width * kernels_shape.height;
            kernel_view.data = &conv->kernels->data[kernel_index];

            ts_tensor_conv_ip(out_temp, &input_view, &kernel_view, conv->stride_x, conv->stride_y);

            ts_tensor_add_ip(&output_view, &output_view, out_temp);
        }
    }

    ts_tensor_add_ip(output, output, conv->biases);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    // Biases change is just delta
    ts_param_change_add(&conv->biases_change, delta);

    ts_tensor* input = ts_layers_cache_pop(cache);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_tensor* kernels_change = ts_tensor_create(scratch.arena, conv->kernels->shape);
    ts_tensor* delta_out = ts_tensor_create(scratch.arena, input->shape);

    ts_tensor input_view = { 0 };
    ts_tensor delta_out_view = { 0 };
    ts_tensor delta_view = { 0 };

    // Stores individual kernel and kernel change of each iteration
    ts_tensor kernel_view = { .shape = conv->kernel_size };
    ts_tensor kernel_change_view = { .shape = conv->kernel_size };
    ts_tensor_shape kernels_shape = conv->kernels->shape;

    // Input and Delta out pos: i_x, i_y, i_z
    // Delta pos: d_x, d_y, d_z
    // Kernel pos: k_x, k_y
    for (ts_u32 d_z = 0; d_z < delta->shape.depth; d_z++) {
        ts_tensor_2d_view(&delta_view, delta, d_z);

        for (ts_u32 i_z = 0; i_z < input->shape.depth; i_z++) {
            ts_tensor_2d_view(&input_view, input, i_z);
            ts_tensor_2d_view(&delta_out_view, delta_out, i_z);

            ts_u64 kernel_index = (ts_u64)i_z * kernels_shape.width + (ts_u64)d_z * kernels_shape.width * kernels_shape.height;
            kernel_view.data = &conv->kernels->data[kernel_index];
            kernel_change_view.data = &kernels_change->data[kernel_index];

            for (ts_u32 d_y = 0, i_y = 0; d_y < delta_view.shape.height; d_y++, i_y += conv->stride_y) {
                for (ts_u32 d_x = 0, i_x = 0; d_x < delta_view.shape.width; d_x++, i_x += conv->stride_x) {
                    ts_u64 delta_view_pos = (ts_u64)d_x + (ts_u64)d_y * delta_view.shape.width;

                    ts_f32 cur_orig_delta = delta_view.data[delta_view_pos];

                    for (ts_u32 k_y = 0; k_y < kernel_change_view.shape.height; k_y++) {
                        for (ts_u32 k_x = 0; k_x < kernel_change_view.shape.width; k_x++) {
                            ts_u64 in_pos = (ts_u64)(i_x + k_x) + (ts_u64)(i_y + k_y) * input->shape.width;
                            ts_u64 kernel_pos = (ts_u64)k_x + (ts_u64)k_y * kernel_change_view.shape.width;

                            // Updating kernel_change
                            kernel_change_view.data[kernel_pos] += input_view.data[in_pos] * cur_orig_delta;

                            // Updating delta out
                            delta_out_view.data[in_pos] = cur_orig_delta * kernel_view.data[kernel_pos];
                        }
                    }
                }
            }
        }
    }

    if (ts_tensor_shape_eq(conv->input_shape, conv->padded_shape)) {
        ts_tensor_copy_ip(delta, delta_out);
    } else {
        delta->shape = conv->input_shape;

        ts_u32 x_off = (conv->padded_shape.width - conv->input_shape.width) / 2;
        ts_u32 y_off = (conv->padded_shape.height - conv->input_shape.height) / 2;

        for (ts_u32 z = 0; z < delta->shape.depth; z++) {
            for (ts_u32 y = 0; y < delta->shape.height; y++) {
                for (ts_u32 x = 0; x < delta->shape.width; x++) {
                    ts_u64 delta_pos = (ts_u64)x + 
                        (ts_u64)y * delta->shape.width + 
                        (ts_u64)z * delta->shape.width * delta->shape.height;
                    ts_u64 padded_delta_pos = (ts_u64)(x + x_off) + 
                        (ts_u64)(y + y_off) * delta_out->shape.width + 
                        (ts_u64)z * delta_out->shape.width * delta_out->shape.height;

                    delta->data[delta_pos] = delta_out->data[padded_delta_pos];
                }
            }
        }
    }

    ts_param_change_add(&conv->kernels_change, kernels_change);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_apply_changes(ts_layer* l, const ts_optimizer* optim) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    ts_param_change_update(optim, conv->kernels, &conv->kernels_change);
    ts_param_change_update(optim, conv->biases, &conv->biases_change);
}
void _layer_conv_2d_delete(ts_layer* l) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    if (l->training_mode) {
        ts_param_change_delete(&conv->kernels_change);
        ts_param_change_delete(&conv->biases_change);
    }
}
void _layer_conv_2d_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    ts_string8 kernels_name = ts_str8_pushf(arena, "conv_2d_kernels_%u", index);
    ts_string8 biases_name = ts_str8_pushf(arena, "conv_2d_biases_%u", index);

    ts_tensor_list_push(arena, list, conv->kernels, kernels_name);
    ts_tensor_list_push(arena, list, conv->biases, biases_name);
}
void _layer_conv_2d_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index) {
    _layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 kernels_name = ts_str8_pushf(scratch.arena, "conv_2d_kernels_%u", index);
    ts_string8 biases_name = ts_str8_pushf(scratch.arena, "conv_2d_biases_%u", index);

    ts_tensor* loaded_kernels = ts_tensor_list_get(list, kernels_name);
    ts_tensor* loaded_biases = ts_tensor_list_get(list, biases_name);

    ts_tensor_copy_ip(conv->kernels, loaded_kernels);
    ts_tensor_copy_ip(conv->biases, loaded_biases);

    mga_scratch_release(scratch);
}

