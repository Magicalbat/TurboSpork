#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>

/*static _layer_create_func* create_funcs[LAYER_COUNT] = {
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
};*/

#define _DEF_LAYER_FUNCS(layer_name) \
    { \
        _layer_##layer_name##_create, \
        _layer_##layer_name##_feedforward, \
        _layer_##layer_name##_backprop, \
        _layer_##layer_name##_apply_changes, \
    }

static _layer_func_defs layer_funcs[LAYER_COUNT] = {
    _DEF_LAYER_FUNCS(null),
    _DEF_LAYER_FUNCS(dense),
    _DEF_LAYER_FUNCS(activation),
};

layer* layer_create(mg_arena* arena, const layer_desc* desc) {
    if (desc->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot create layer: invalid type\n");
        return NULL;
    }

    layer* out = MGA_PUSH_ZERO_STRUCT(arena, layer);

    out->type = desc->type;
    out->training_mode = desc->training_mode;

    layer_funcs[desc->type].create(arena, out, desc);

    return out;
}
void layer_feedforward(layer* l, tensor* in_out) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    layer_funcs[l->type].feedforward(l, in_out);
}
void layer_backprop(layer* l, tensor* delta) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    layer_funcs[l->type].backprop(l, delta);
}
void layer_apply_changes(layer* l) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    layer_funcs[l->type].apply_changes(l);
}

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc) {
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

