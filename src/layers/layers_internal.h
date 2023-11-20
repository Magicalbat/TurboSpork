#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"

typedef struct {
    tensor* weight;
    tensor* weight_transposed;
    tensor* bias;

    // Training mode
    param_change weight_change;
    param_change bias_change;
} layer_dense_backend;

typedef struct {
    layer_activation_type type;
} layer_activation_backend;

typedef struct {
    f32 keep_rate;
} layer_dropout_backend;

typedef struct {
    tensor_shape prev_shape;
} layer_flatten_backend;

typedef struct {
    tensor_shape pool_size;

    layer_pooling_type type;
} layer_pooling_2d_backend;

typedef struct layer {
    // Initialized in layer_create
    layer_type type;
    b32 training_mode;

    // Should be set by layer
    tensor_shape shape;

    union {
        layer_dense_backend dense_backend;
        layer_activation_backend activation_backend;
        layer_dropout_backend dropout_backend;
        layer_flatten_backend flatten_backend;
        layer_pooling_2d_backend pooling_2d_backend;
    };
} layer;

void param_init(tensor* param, param_init_type input_type, param_init_type default_type, u32 in_size, u32 out_size);

// TODO: consistent underscoring for private stuff
typedef struct layers_cache_node {
    tensor* t;
    struct layers_cache_node* next;
} layers_cache_node;

typedef struct layers_cache {
    mg_arena* arena;

    layers_cache_node* first;
    layers_cache_node* last;
} layers_cache;

void layers_cache_push(layers_cache* cache, tensor* t);
tensor* layers_cache_pop(layers_cache* cache);

typedef void (_layer_create_func)(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
typedef void (_layer_feedforward_func)(layer* l, tensor* in_out, layers_cache* cache);
typedef void (_layer_backprop_func)(layer* l, tensor* delta, layers_cache* cache); typedef void (_layer_apply_changes_func)(layer* l, const optimizer* optim);
typedef void (_layer_delete_func)(layer* l);
typedef void (_layer_save_func)(mg_arena* arena, tensor_list* list, layer* l, u32 index);
typedef void (_layer_load_func)(layer* l, const tensor_list* list, u32 index);

typedef struct {
    _layer_create_func* create;
    _layer_feedforward_func* feedforward;
    _layer_backprop_func* backprop;
    _layer_apply_changes_func* apply_changes;
    _layer_delete_func* delete;
    _layer_save_func* save;
    _layer_load_func* load;
} _layer_func_defs;

// These functions are implemented in layers.c or a specific layers_*.c file

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_null_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_null_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_null_apply_changes(layer* l, const optimizer* optim);
void _layer_null_delete(layer* l);
void _layer_null_save(mg_arena* arena, tensor_list* list, layer* l, u32 index);
void _layer_null_load(layer* l, const tensor_list* list, u32 index);

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape); 

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dense_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dense_backprop(layer* l, tensor* delta, layers_cache* cache);
void _layer_dense_apply_changes(layer* l, const optimizer* optim);
void _layer_dense_delete(layer* l);
void _layer_dense_save(mg_arena* arena, tensor_list* list, layer* l, u32 index);
void _layer_dense_load(layer* l, const tensor_list* list, u32 index);

void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_activation_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_activation_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_dropout_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_dropout_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_dropout_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_flatten_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_flatten_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_flatten_backprop(layer* l, tensor* delta, layers_cache* cache);

void _layer_pooling_2d_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape);
void _layer_pooling_2d_feedforward(layer* l, tensor* in_out, layers_cache* cache);
void _layer_pooling_2d_backprop(layer* l, tensor* delta, layers_cache* cache);

#endif // LAYERS_INTERNAL_H

