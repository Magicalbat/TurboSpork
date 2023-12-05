#include "layers.h"
#include "layers_internal.h"

void _layer_null_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);
    TS_UNUSED(desc);

    out->shape = prev_shape;
}
void _layer_null_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    TS_UNUSED(l);
    TS_UNUSED(in_out);
    TS_UNUSED(cache);
}
void _layer_null_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    TS_UNUSED(l);
    TS_UNUSED(delta);
    TS_UNUSED(cache);
}
void _layer_null_apply_changes(ts_layer* l, const ts_optimizer* optim) {
    TS_UNUSED(l);
    TS_UNUSED(optim);
}
void _layer_null_delete(ts_layer* l) {
    TS_UNUSED(l);
}
void _layer_null_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index) {
    TS_UNUSED(arena);
    TS_UNUSED(list);
    TS_UNUSED(l);
    TS_UNUSED(index);
}
void _layer_null_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index) {
    TS_UNUSED(l);
    TS_UNUSED(list);
    TS_UNUSED(index);
}
