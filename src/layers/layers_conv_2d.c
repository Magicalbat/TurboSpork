#include "layers.h"
#include "layers_internal.h"

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

    out->shape = conv->padded_shape;
    out->shape.depth = cdesc->num_filters;

    // Shrinking out shape based on kernel size and stride
    out->shape.width -= (conv->kernel_size.width - 1);
    out->shape.width -= (conv->stride_x - 1);
    out->shape.height -= (conv->kernel_size.height - 1);
    out->shape.height -= (conv->stride_y - 1);

    // Have to collapse one dimension because tensors are only 3d
    tensor_shape kernels_shape = {
        .width = conv->kernel_size.width * conv->kernel_size.height,
        .height = prev_shape.depth,
        .depth = cdesc->num_filters
    };

    conv->kernels = tensor_create(arena, kernels_shape);
    conv->biases = tensor_create(arena, out->shape);

    // TODO: figure out in and out sizes for the param init
    param_init(conv->kernels, cdesc->kernels_init, 0, 1);
    param_init(conv->biases, cdesc->biases_init, 0, 1);

    param_change_create(arena, &conv->kernels_change, kernels_shape);
    param_change_create(arena, &conv->biases_change, out->shape);
}
void _layer_conv_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache) {}
void _layer_conv_2d_backprop(layer* l, tensor* delta, layers_cache* cache) {}
void _layer_conv_2d_apply_changes(layer* l, const optimizer* optim) {}
void _layer_conv_2d_delete(layer* l) {
    layer_conv_2d_backend* conv = &l->conv_2d_backend;

    param_change_delete(&conv->kernels_change);
    param_change_delete(&conv->biases_change);
}
void _layer_conv_2d_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {}
void _layer_conv_2d_load(layer* l, const tensor_list* list, u32 index) {}


