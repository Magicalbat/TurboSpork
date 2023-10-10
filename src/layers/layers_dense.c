#include "layers.h"
#include "layers_internal.h"

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc) {
    u32 in_size = desc->dense.in_size;
    u32 out_size = desc->dense.out_size;

    out->type = LAYER_DENSE;

    tensor_shape bias_shape = { out_size, 1, 1 };
    tensor_shape weight_shape = { out_size, in_size, 1 };

    out->input_shape = (tensor_shape){ in_size, 1, 1 };
    out->output_shape = bias_shape;

    layer_dense_backend* dense = &out->dense_backend;

    dense->bias = tensor_create(arena, bias_shape);
    dense->bias_change = tensor_create(arena, bias_shape);
    dense->weight = tensor_create(arena, weight_shape);
    dense->weight_change = tensor_create(arena, weight_shape);

    // TODO: weight init
}
void _layer_dense_feedforward(layer* l, tensor* in_out) {
    layer_dense_backend* dense = &l->dense_backend;

    tensor_dot_ip(in_out, in_out, dense->weight);
    tensor_add_ip(in_out, in_out, dense->bias);
}
void _layer_dense_backprop(layer* l, tensor* delta) {
    // TODO
}
void _layer_dense_apply_changes(layer* l) {
    layer_dense_backend* dense = &l->dense_backend;

    tensor_add_ip(dense->weight, dense->weight, dense->weight_change);
    tensor_add_ip(dense->bias, dense->bias, dense->bias_change);

    tensor_fill(dense->weight_change, 0.0f);
    tensor_fill(dense->bias_change, 0.0f);
}

