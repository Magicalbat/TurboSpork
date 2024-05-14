#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"


// TODO: consistent underscoring for private stuff
typedef struct {
    ts_layer_create_func* create;
    ts_layer_feedforward_func* feedforward;
    ts_layer_backprop_func* backprop;
    ts_layer_apply_changes_func* apply_changes;
    ts_layer_delete_func* delete;
    ts_layer_save_func* save;
    ts_layer_load_func* load;
} _layer_func_defs;

// These functions are implemented in specific layers_*.c files

void _layer_null_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_null_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_null_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
void _layer_null_apply_changes(ts_layer* l, const ts_optimizer* optim);
void _layer_null_delete(ts_layer* l);
void _layer_null_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index);
void _layer_null_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

void _layer_input_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape); 
void _layer_input_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);

void _layer_reshape_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_reshape_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_reshape_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

void _layer_dense_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_dense_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_dense_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
void _layer_dense_apply_changes(ts_layer* l, const ts_optimizer* optim);
void _layer_dense_delete(ts_layer* l);
void _layer_dense_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index);
void _layer_dense_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

void _layer_activation_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_activation_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_activation_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

void _layer_dropout_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_dropout_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_dropout_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

void _layer_flatten_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_flatten_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_flatten_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

void _layer_pooling_2d_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_pooling_2d_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_pooling_2d_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

void _layer_conv_2d_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_conv_2d_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_conv_2d_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
void _layer_conv_2d_apply_changes(ts_layer* l, const ts_optimizer* optim);
void _layer_conv_2d_delete(ts_layer* l);
void _layer_conv_2d_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index);
void _layer_conv_2d_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

void _layer_norm_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_norm_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_norm_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);

#endif // LAYERS_INTERNAL_H

