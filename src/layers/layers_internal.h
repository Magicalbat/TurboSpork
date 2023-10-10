#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"

typedef void (_layer_create_func)(mg_arena* arena, layer* out, const layer_desc* desc);
typedef void (_layer_feedforward_func)(layer* l, tensor* in_out);
typedef void (_layer_backprop_func)(layer* l, tensor* delta);
typedef void (_layer_apply_changes_func)(layer* l);

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc);
void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc);
void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc);

void _layer_null_feedforward(layer* l, tensor* in_out);
void _layer_dense_feedforward(layer* l, tensor* in_out);
void _layer_activation_feedforward(layer* l, tensor* in_out);

void _layer_null_backprop(layer* l, tensor* delta);
void _layer_dense_backprop(layer* l, tensor* delta);
void _layer_activation_backprop(layer* l, tensor* delta);

void _layer_null_apply_changes(layer* l);
void _layer_dense_apply_changes(layer* l);
void _layer_activation_apply_changes(layer* l);

#endif // LAYERS_INTERNAL_H

