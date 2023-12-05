#include "layers.h"
#include "layers_internal.h"

void _layer_input_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);
    TS_UNUSED(prev_shape);

    out->shape = desc->input.shape;
}
void _layer_input_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    TS_UNUSED(cache);

    in_out->shape = l->shape;
}

