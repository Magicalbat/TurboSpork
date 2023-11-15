#include "layers.h"
#include "layers_internal.h"

void _layer_flatten_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(desc);

    out->flatten_backend.prev_shape = prev_shape;
    u64 out_size = (u64)prev_shape.width * prev_shape.height * prev_shape.depth;

    out->shape = (tensor_shape){ out_size, 1, 1 };
}
void _layer_flatten_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    UNUSED(cache);

    in_out->shape = l->shape;
}
void _layer_flatten_backprop(layer* l, tensor* delta, layers_cache* cache) {
    UNUSED(cache);

    delta->shape = l->flatten_backend.prev_shape;
}

