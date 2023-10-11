#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"

typedef void (_layer_create_func)(mg_arena* arena, layer* out, const layer_desc* desc);
typedef void (_layer_feedforward_func)(layer* l, tensor* in_out);
typedef void (_layer_backprop_func)(layer* l, tensor* delta);
typedef void (_layer_apply_changes_func)(layer* l, u32 batch_size);

typedef struct {
    _layer_create_func* create;
    _layer_feedforward_func* feedforward;
    _layer_backprop_func* backprop;
    _layer_apply_changes_func* apply_changes;
} _layer_func_defs;

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc);
void _layer_null_feedforward(layer* l, tensor* in_out);
void _layer_null_backprop(layer* l, tensor* delta);
void _layer_null_apply_changes(layer* l, u32 batch_size);

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc);
void _layer_dense_feedforward(layer* l, tensor* in_out);
void _layer_dense_backprop(layer* l, tensor* delta);
void _layer_dense_apply_changes(layer* l, u32 batch_size);

void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc);
void _layer_activation_feedforward(layer* l, tensor* in_out);
void _layer_activation_backprop(layer* l, tensor* delta);
void _layer_activation_apply_changes(layer* l, u32 batch_size);

#endif // LAYERS_INTERNAL_H

