#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>

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

    if (out->training_mode) {
        out->prev_input = tensor_create(arena, out->input_shape);
    }

    return out;
}
void layer_feedforward(layer* l, tensor* in_out) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    if (l->training_mode) {
        tensor_copy_ip(l->prev_input, in_out);
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
void layer_apply_changes(layer* l, u32 batch_size) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    layer_funcs[l->type].apply_changes(l, batch_size);
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
void _layer_null_apply_changes(layer* l, u32 batch_size) {
    UNUSED(l);
    UNUSED(batch_size);
}

