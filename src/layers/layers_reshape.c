#include "layers.h"
#include "layers_internal.h"

void _layer_reshape_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);

    out->shape = desc->reshape.shape;

    out->reshape_backend.prev_shape = prev_shape;
}
void _layer_reshape_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    UNUSED(cache);

    in_out->shape = l->shape;
}
void _layer_reshape_backprop(layer* l, tensor* delta, layers_cache* cache) {
    UNUSED(cache);

    delta->shape = l->reshape_backend.prev_shape;
}

