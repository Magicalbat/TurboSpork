#include "layers.h"
#include "layers_internal.h"

static _layer_create_func* create_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = _layer_null_create,
    [LAYER_DENSE] = _layer_dense_create,
    [LAYER_ACTIVATION] = _layer_activation_create,
};
static _layer_feedforward_func* feedforward_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = _layer_null_feedforward,
    [LAYER_DENSE] = _layer_dense_feedforward,
    [LAYER_ACTIVATION] = _layer_activation_feedforward,
};
static _layer_backprop_func* backprop_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = _layer_null_backprop,
    [LAYER_DENSE] = _layer_dense_backprop,
    [LAYER_ACTIVATION] = _layer_activation_backprop,
};
static _layer_apply_changes_func* apply_changes_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = _layer_null_apply_changes,
    [LAYER_DENSE] = _layer_dense_apply_changes,
    [LAYER_ACTIVATION] = _layer_activation_apply_changes,
};

void _layer_null_create(mg_arena* arena, layer* out, layer_desc* desc) {
    UNUSED(arena);
    UNUSED(out);
    UNUSED(desc);
}
void _layer_null_feedforward(layer* l, tensor* in_out) {
    UNUSED(l);
    UNUSED(in_out);
}
void _layer_null_backprop(layer* l, tensor* delta) {
    UNUSED(l);
    UNUSED(delta);
}
void _layer_null_apply_changes(layer* l) {
    UNUSED(l);
}

