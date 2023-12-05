#include "layers.h"
#include "layers_internal.h"

void _layer_reshape_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    out->shape = desc->reshape.shape;

    out->reshape_backend.prev_shape = prev_shape;
}
void _layer_reshape_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    TS_UNUSED(cache);

    in_out->shape = l->shape;
}
void _layer_reshape_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    TS_UNUSED(cache);

    delta->shape = l->reshape_backend.prev_shape;
}

