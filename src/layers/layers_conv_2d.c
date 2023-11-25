#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>
void _layer_conv_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    const layer_conv_2d_desc* cdesc = &desc->conv_2d;
    layer_conv_2d_backend* conv = &out->conv_2d_backend;

    conv->kernel_size = cdesc->kernel_size;
    conv->stride_x = cdesc->stride_x;
    conv->stride_y = cdesc->stride_y;

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

    param_change_create(arena, &conv->kernels_change, kernels_shape);
    param_change_create(arena, &conv->biases_change, out->shape);
}
void _layer_conv_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    mga_temp scratch = mga_scratch_get(&cache->arena, 1);

    tensor* input = in_out;
    if (!tensor_shape_eq(in_out->shape, conv->padded_shape)){
        // Create padded shape

        input = tensor_create(scratch.arena, conv->padded_shape);

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

    mga_scratch_release(scratch);
}
void _layer_conv_2d_backprop(layer* l, tensor* delta, layers_cache* cache) {}
void _layer_conv_2d_apply_changes(layer* l, const optimizer* optim) {}
void _layer_conv_2d_delete(layer* l) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    param_change_delete(&conv->kernels_change);
    param_change_delete(&conv->biases_change);
}
void _layer_conv_2d_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {}
void _layer_conv_2d_load(layer* l, const tensor_list* list, u32 index) {}


