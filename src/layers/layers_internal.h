#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"

typedef void (_layer_create_func)(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
typedef void (_layer_feedforward_func)(layer* l, tensor* in_out, layers_cache* cache);
typedef void (_layer_backprop_func)(layer* l, tensor* delta, layers_cache* cache); typedef void (_layer_apply_changes_func)(layer* l, const optimizer* optim);
typedef void (_layer_delete_func)(layer* l);

typedef struct {
    _layer_create_func* create;
    _layer_feedforward_func* feedforward;
    _layer_backprop_func* backprop;
    _layer_apply_changes_func* apply_changes;
    _layer_delete_func* delete;
} _layer_func_defs;

// These functions are implemented in layers.c or a specific layers_*.c file

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_null_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_null_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_null_apply_changes(layer* l, const optimizer* optim);
void _layer_null_delete(layer* l);

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape); 
// Uses null for the rest

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dense_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dense_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_dense_apply_changes(layer* l, const optimizer* optim);
void _layer_dense_delete(layer* l);

void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_activation_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_activation_backprop(layer* l, tensor* delta, layers_cache* cache);
// Uses null apply_changes and delete

void _layer_dropout_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dropout_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dropout_backprop(layer* l, tensor* delta, layers_cache* cache);
// Uses null apply_changes and delete

#endif // LAYERS_INTERNAL_H

