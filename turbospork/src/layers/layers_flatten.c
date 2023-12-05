#include "layers.h"
#include "layers_internal.h"

void _layer_flatten_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);
    TS_UNUSED(desc);

    out->flatten_backend.prev_shape = prev_shape;
    ts_u64 out_size = (ts_u64)prev_shape.width * prev_shape.height * prev_shape.depth;

    out->shape = (ts_tensor_shape){ out_size, 1, 1 };
}
void _layer_flatten_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    TS_UNUSED(cache);

    in_out->shape = l->shape;
}
void _layer_flatten_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    TS_UNUSED(cache);

    delta->shape = l->flatten_backend.prev_shape;
}

