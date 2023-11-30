#include "layers.h"
#include "layers_internal.h"

void _layer_conv_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    const layer_conv_2d_desc* cdesc = &desc->conv_2d;
    layer_conv_2d_backend* conv = &out->conv_2d_backend;

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

    out->shape = tensor_conv_shape(conv->padded_shape, conv->kernel_size, conv->stride_x, conv->stride_y);
    out->shape.depth = cdesc->num_filters;

    // Have to collapse one dimension because tensors are only 3d
    tensor_shape kernels_shape = {
        .width = conv->kernel_size.width * conv->kernel_size.height,
        .height = prev_shape.depth,
        .depth = cdesc->num_filters
    };

    conv->kernels = tensor_create(arena, kernels_shape);
    conv->biases = tensor_create(arena, out->shape);

    u64 in_size = (u64)prev_shape.width * prev_shape.height * prev_shape.depth;
    u64 out_size = (u64)out->shape.width * out->shape.height * out->shape.depth;
    param_init(conv->kernels, cdesc->kernels_init, in_size, out_size);
    param_init(conv->biases, cdesc->biases_init, in_size, out_size);

    if (desc->training_mode) { 
        param_change_create(arena, &conv->kernels_change, kernels_shape);
        param_change_create(arena, &conv->biases_change, out->shape);
    }
}
void _layer_conv_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = { 0 };
    mg_arena* input_arena = NULL;
    if (cache != NULL) {
        scratch = mga_scratch_get(&cache->arena, 1);
        input_arena = cache->arena;
    } else {
        scratch = mga_scratch_get(NULL, 0);
        input_arena = scratch.arena;
    }

    tensor* input = NULL;
    if (tensor_shape_eq(in_out->shape, conv->padded_shape)){
        input = tensor_copy(input_arena, in_out, false);
    } else {
        // Create padded shape
        input = tensor_create(input_arena, conv->padded_shape);

        u32 x_off = (conv->padded_shape.width - in_out->shape.width) / 2;
        u32 y_off = (conv->padded_shape.height - in_out->shape.height) / 2;

        for (u32 z = 0; z < in_out->shape.depth; z++) {
            for (u32 y = 0; y < in_out->shape.height; y++) {
                for (u32 x = 0; x < in_out->shape.width; x++) {
                    u64 in_out_index = (u64)x +
                        (u64)y * in_out->shape.width +
                        (u64)z * in_out->shape.width * in_out->shape.height;
                    u64 input_index = (u64)x + x_off +
                        (u64)(y + y_off) * conv->padded_shape.width +
                        (u64)z * conv->padded_shape.width * conv->padded_shape.height;

                    input->data[input_index] = in_out->data[in_out_index];
                }
            }
        }
    }

    if (cache != NULL) {
        layers_cache_push(cache, input);
    }

    // Renaming for clarity
    tensor* output = in_out;

    output->shape = l->shape;
    tensor_fill(output, 0.0f);

    tensor input_view = { 0 };
    tensor output_view = { 0 };

    // Stores individual kernel of each iteration
    tensor kernel_view = { .shape = conv->kernel_size };
    tensor_shape kernels_shape = conv->kernels->shape;

    // Used for storing a conv output before adding to output
    tensor* out_temp = tensor_create(scratch.arena, (tensor_shape){ output->shape.width, output->shape.height, 1 });

    for (u32 o_z = 0; o_z < output->shape.depth; o_z++) {
        tensor_2d_view(&output_view, output, o_z);

        for (u32 i_z = 0; i_z < input->shape.depth; i_z++) {
            tensor_2d_view(&input_view, input, i_z);

            u64 kernel_index = (u64)i_z * kernels_shape.width + (u64)o_z * kernels_shape.width * kernels_shape.height;
            kernel_view.data = &conv->kernels->data[kernel_index];

            tensor_conv_ip(out_temp, &input_view, &kernel_view, conv->stride_x, conv->stride_y);

            tensor_add_ip(&output_view, &output_view, out_temp);
        }
    }

    tensor_add_ip(output, output, conv->biases);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_backprop(layer* l, tensor* delta, layers_cache* cache) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    // Biases change is just delta
    param_change_add(&conv->biases_change, delta);

    tensor* input = layers_cache_pop(cache);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor* kernels_change = tensor_create(scratch.arena, conv->kernels->shape);
    tensor* delta_out = tensor_create(scratch.arena, input->shape);

    tensor input_view = { 0 };
    tensor delta_out_view = { 0 };
    tensor delta_view = { 0 };

    // Stores individual kernel and kernel change of each iteration
    tensor kernel_view = { .shape = conv->kernel_size };
    tensor kernel_change_view = { .shape = conv->kernel_size };
    tensor_shape kernels_shape = conv->kernels->shape;

    // Input and Delta out pos: i_x, i_y, i_z
    // Delta pos: d_x, d_y, d_z
    // Kernel pos: k_x, k_y
    for (u32 d_z = 0; d_z < delta->shape.depth; d_z++) {
        tensor_2d_view(&delta_view, delta, d_z);

        for (u32 i_z = 0; i_z < input->shape.depth; i_z++) {
            tensor_2d_view(&input_view, input, i_z);
            tensor_2d_view(&delta_out_view, delta_out, i_z);

            u64 kernel_index = (u64)i_z * kernels_shape.width + (u64)d_z * kernels_shape.width * kernels_shape.height;
            kernel_view.data = &conv->kernels->data[kernel_index];
            kernel_change_view.data = &kernels_change->data[kernel_index];

            for (u32 d_y = 0, i_y = 0; d_y < delta_view.shape.height; d_y++, i_y += conv->stride_y) {
                for (u32 d_x = 0, i_x = 0; d_x < delta_view.shape.width; d_x++, i_x += conv->stride_x) {
                    u64 delta_view_pos = (u64)d_x + (u64)d_y * delta_view.shape.width;

                    f32 cur_orig_delta = delta_view.data[delta_view_pos];

                    for (u32 k_y = 0; k_y < kernel_change_view.shape.height; k_y++) {
                        for (u32 k_x = 0; k_x < kernel_change_view.shape.width; k_x++) {
                            u64 in_pos = (u64)(i_x + k_x) + (u64)(i_y + k_y) * input->shape.width;
                            u64 kernel_pos = (u64)k_x + (u64)k_y * kernel_change_view.shape.width;

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

    if (tensor_shape_eq(conv->input_shape, conv->padded_shape)) {
        tensor_copy_ip(delta, delta_out);
    } else {
        delta->shape = conv->input_shape;

        u32 x_off = (conv->padded_shape.width - conv->input_shape.width) / 2;
        u32 y_off = (conv->padded_shape.height - conv->input_shape.height) / 2;

        for (u32 z = 0; z < delta->shape.depth; z++) {
            for (u32 y = 0; y < delta->shape.height; y++) {
                for (u32 x = 0; x < delta->shape.width; x++) {
                    u64 delta_pos = (u64)x + 
                        (u64)y * delta->shape.width + 
                        (u64)z * delta->shape.width * delta->shape.height;
                    u64 padded_delta_pos = (u64)(x + x_off) + 
                        (u64)(y + y_off) * delta_out->shape.width + 
                        (u64)z * delta_out->shape.width * delta_out->shape.height;

                    delta->data[delta_pos] = delta_out->data[padded_delta_pos];
                }
            }
        }
    }

    param_change_add(&conv->kernels_change, kernels_change);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_apply_changes(layer* l, const optimizer* optim) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    param_change_update(optim, conv->kernels, &conv->kernels_change);
    param_change_update(optim, conv->biases, &conv->biases_change);
}
void _layer_conv_2d_delete(layer* l) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    param_change_delete(&conv->kernels_change);
    param_change_delete(&conv->biases_change);
}
void _layer_conv_2d_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    string8 kernels_name = str8_pushf(arena, "conv_2d_kernels_%u", index);
    string8 biases_name = str8_pushf(arena, "conv_2d_biases_%u", index);

    tensor_list_push(arena, list, conv->kernels, kernels_name);
    tensor_list_push(arena, list, conv->biases, biases_name);
}
void _layer_conv_2d_load(layer* l, const tensor_list* list, u32 index) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 kernels_name = str8_pushf(scratch.arena, "conv_2d_kernels_%u", index);
    string8 biases_name = str8_pushf(scratch.arena, "conv_2d_biases_%u", index);

    tensor* loaded_kernels = tensor_list_get(list, kernels_name);
    tensor* loaded_biases = tensor_list_get(list, biases_name);

    tensor_copy_ip(conv->kernels, loaded_kernels);
    tensor_copy_ip(conv->biases, loaded_biases);

    mga_scratch_release(scratch);
}

