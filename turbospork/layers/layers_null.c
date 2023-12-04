#include "layers.h"
#include "layers_internal.h"

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(desc);

    out->shape = prev_shape;
}
void _layer_null_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    UNUSED(l);
    UNUSED(in_out);
    UNUSED(cache);
}
void _layer_null_backprop(layer* l, tensor* delta, layers_cache* cache) {
    UNUSED(l);
    UNUSED(delta);
    UNUSED(cache);
}
void _layer_null_apply_changes(layer* l, const optimizer* optim) {
    UNUSED(l);
    UNUSED(optim);
}
void _layer_null_delete(layer* l) {
    UNUSED(l);
}
void _layer_null_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {
    UNUSED(arena);
    UNUSED(list);
    UNUSED(l);
    UNUSED(index);
}
void _layer_null_load(layer* l, const tensor_list* list, u32 index) {
    UNUSED(l);
    UNUSED(list);
    UNUSED(index);
}
