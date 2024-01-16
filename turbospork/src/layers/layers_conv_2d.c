#include "layers.h"
#include "layers_internal.h"

void _layer_conv_2d_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    const ts_layer_conv_2d_desc* cdesc = &desc->conv_2d;
    ts_layer_conv_2d_backend* conv = &out->conv_2d_backend;

    conv->kernel_size = cdesc->kernel_size;
    conv->stride = cdesc->stride;
    conv->input_shape = prev_shape;

    if (cdesc->padding) {
        // Padding so that out_shape == in_shape when stride is 1
        conv->padding = (conv->kernel_size - 1) / 2;
    }

    ts_tensor_shape padded_shape = (ts_tensor_shape){
        prev_shape.width + conv->padding * 2,
        prev_shape.height + conv->padding * 2,
        prev_shape.depth
    };
    out->shape = ts_tensor_conv_shape(padded_shape, (ts_tensor_shape){ conv->kernel_size, conv->kernel_size, 1 }, conv->stride, conv->stride);
    out->shape.depth = cdesc->num_filters;

    // Have to collapse one dimension because ts_tensors are only 3d
    ts_tensor_shape kernels_shape = {
        .width = conv->kernel_size * conv->kernel_size * prev_shape.depth,
        .height = cdesc->num_filters,
        .depth = 1
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
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = { 0 };
    mg_arena* col_arena = NULL;
    if (cache != NULL) {
        col_arena = cache->arena;
    } else {
        scratch = mga_scratch_get(NULL, 0);
        col_arena = scratch.arena;
    }

    // Article explaining how to turn conv into mat mul
    // https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/

    // Renaming for clarity
    ts_tensor* output = in_out;

    ts_tensor* input_cols = ts_tensor_im2col(col_arena, in_out, conv->kernel_size, conv->stride, conv->padding);
    if (cache != NULL) {
        ts_layers_cache_push(cache, input_cols);
    }

    ts_tensor_dot_ip(output, false, false, conv->kernels, input_cols);

    output->shape = l->shape;

    ts_tensor_add_ip(output, output, conv->biases);

    if (scratch.arena != NULL) {
        mga_scratch_release(scratch);
    }
}

void _layer_conv_2d_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    // Biases change is just delta
    ts_param_change_add(&conv->biases_change, delta);

    ts_tensor* input_cols = ts_layers_cache_pop(cache);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Updating kernels
    // kernel_change = delta * prev_input
    ts_tensor delta_view = (ts_tensor){
        .shape = (ts_tensor_shape){
            delta->shape.width * delta->shape.height,
            delta->shape.depth, 1
        },
        .data = delta->data
    };
    
    ts_tensor* kernels_change = ts_tensor_create(
        scratch.arena,
        (ts_tensor_shape){
            conv->kernels->shape.width * conv->kernels->shape.height,
            conv->kernels->shape.depth, 1
        }
    );

    
    ts_tensor_dot_ip(kernels_change, false, true, &delta_view, input_cols);
    ts_param_change_add(&conv->kernels_change, kernels_change);

    // Resetting scratch after kernels change
    mga_temp_end(scratch);

    // Updating delta
    // delta *= kernels
    // Math is done in columns, then converted back to an image

    ts_tensor* delta_cols = ts_tensor_dot(scratch.arena, true, false, conv->kernels, &delta_view);
    ts_tensor_col2im_ip(delta, delta_cols, conv->input_shape, conv->kernel_size, conv->stride, conv->padding);

    mga_scratch_release(scratch);
}
void _layer_conv_2d_apply_changes(ts_layer* l, const ts_optimizer* optim) {
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    ts_param_change_apply(optim, conv->kernels, &conv->kernels_change);
    ts_param_change_apply(optim, conv->biases, &conv->biases_change);
}
void _layer_conv_2d_delete(ts_layer* l) {
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    if (l->training_mode) {
        ts_param_change_delete(&conv->kernels_change);
        ts_param_change_delete(&conv->biases_change);
    }
}
void _layer_conv_2d_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index) {
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    ts_string8 kernels_name = ts_str8_pushf(arena, "conv_2d_kernels_%u", index);
    ts_string8 biases_name = ts_str8_pushf(arena, "conv_2d_biases_%u", index);

    ts_tensor_list_push(arena, list, conv->kernels, kernels_name);
    ts_tensor_list_push(arena, list, conv->biases, biases_name);
}
void _layer_conv_2d_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index) {
    ts_layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 kernels_name = ts_str8_pushf(scratch.arena, "conv_2d_kernels_%u", index);
    ts_string8 biases_name = ts_str8_pushf(scratch.arena, "conv_2d_biases_%u", index);

    ts_tensor* loaded_kernels = ts_tensor_list_get(list, kernels_name);
    ts_tensor* loaded_biases = ts_tensor_list_get(list, biases_name);

    ts_tensor_copy_ip(conv->kernels, loaded_kernels);
    ts_tensor_copy_ip(conv->biases, loaded_biases);

    mga_scratch_release(scratch);
}

