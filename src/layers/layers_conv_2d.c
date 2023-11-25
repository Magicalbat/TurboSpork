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

    mga_temp scratch = { 0 };
    if (cache != NULL) {
        scratch = mga_scratch_get(&cache->arena, 1);
    } else {
        scratch = mga_scratch_get(NULL, 0);
    }

    tensor* input = NULL;
    if (tensor_shape_eq(in_out->shape, conv->padded_shape)){
        input = tensor_copy(scratch.arena, in_out, false);
    } else {
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

    for (u32 o_w = 0; o_w < output->shape.depth; o_w++) {
        tensor_2d_view(&output_view, output, o_w);

        for (u32 i_w = 0; i_w < input->shape.depth; i_w++) {
            tensor_2d_view(&input_view, input, i_w);

            u64 kernel_index = (u64)i_w * kernels_shape.width + (u64)o_w * kernels_shape.width * kernels_shape.height;
            kernel_view.data = &conv->kernels->data[kernel_index];

            tensor_conv_ip(out_temp, &input_view, &kernel_view, conv->stride_x, conv->stride_y);

            tensor_add_ip(&output_view, &output_view, out_temp);
        }
    }

    tensor_add_ip(output, output, conv->biases);

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


