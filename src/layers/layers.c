#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>

static const char* _layer_names[LAYER_COUNT] = {
    [LAYER_NULL] = "null",
    [LAYER_INPUT] = "input",
    [LAYER_DENSE] = "dense",
    [LAYER_ACTIVATION] = "activation",
};

string8 layer_get_name(layer_type type) {
    if (type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot get name: invalid layer type\n");

        return (string8){ 0 };
    }

    return str8_from_cstr((u8*)_layer_names[type]);
}
layer_type layer_from_name(string8 name) {
    for (layer_type type = LAYER_NULL; type < LAYER_COUNT; type++) {
        if (str8_equals(name, layer_get_name(type))) {
            return type;
        }
    }

    return LAYER_NULL;
}

static _layer_func_defs _layer_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = {
        _layer_null_create,
        _layer_null_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
    },
    [LAYER_INPUT] = {
        _layer_input_create,
        _layer_null_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
    },
    [LAYER_DENSE] = {
        _layer_dense_create,
        _layer_dense_feedforward,
        _layer_dense_backprop,
        _layer_dense_apply_changes,
        _layer_dense_delete,
    },
    [LAYER_ACTIVATION] = {
        _layer_activation_create,
        _layer_activation_feedforward,
        _layer_activation_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
    }
};

layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape) {
    if (desc->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot create layer: invalid type\n");
        return NULL;
    }

    layer* out = MGA_PUSH_ZERO_STRUCT(arena, layer);

    out->type = desc->type;
    out->training_mode = desc->training_mode;

    _layer_funcs[desc->type].create(arena, out, desc, prev_shape);

    return out;
}
void layer_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    _layer_funcs[l->type].feedforward(l, in_out, cache);
}
void layer_backprop(layer* l, tensor* delta, layers_cache* cache) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    _layer_funcs[l->type].backprop(l, delta, cache);
}
void layer_apply_changes(layer* l, const optimizer* optim) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot feedforward layer: invalid type\n");
        return;
    }

    _layer_funcs[l->type].apply_changes(l, optim);
}
void layer_delete(layer* l) {
    if (l->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot delete layer: invalid type\n");
        return;
    }

    _layer_funcs[l->type].delete(l);
}

void layers_cache_push(layers_cache* cache, tensor* t) {
    layers_cache_node* node = MGA_PUSH_ZERO_STRUCT(cache->arena, layers_cache_node);
    node->t = t;

    SLL_PUSH_FRONT(cache->first, cache->last, node);
}
tensor* layers_cache_pop(layers_cache* cache) {
    tensor* out = cache->first->t;

    SLL_POP_FRONT(cache->first, cache->last);

    return out;
}

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

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(prev_shape);

    out->shape = desc->input.shape;

    // Input layer never needs to be in training mode
    out->training_mode = false;
}

