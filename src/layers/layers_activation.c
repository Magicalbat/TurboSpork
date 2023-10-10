#include "layers_internal.h"


void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc) {
    UNUSED(arena);
    UNUSED(out);
    UNUSED(desc);
}
void _layer_activation_feedforward(layer* l, tensor* in_out) {
    UNUSED(l);
    UNUSED(in_out);
}
void _layer_activation_backprop(layer* l, tensor* delta) {
    UNUSED(l);
    UNUSED(delta);
}
void _layer_activation_apply_changes(layer* l) {
    UNUSED(l);
}
