#ifndef LAYERS_INTERNAL_H
#define LAYERS_INTERNAL_H

#include "layers.h"

typedef struct {
    ts_tensor_shape prev_shape;
} _layer_reshape_backend;

typedef struct {
    ts_tensor* weight;
    ts_tensor* weight_transposed;
    ts_tensor* bias;

    // Training mode
    ts_param_change weight_change;
    ts_param_change bias_change;
} _layer_dense_backend;

typedef struct {
    ts_layer_activation_type type;
} _layer_activation_backend;

typedef struct {
    ts_f32 keep_rate;
} _layer_dropout_backend;

typedef struct {
    ts_tensor_shape prev_shape;
} _layer_flatten_backend;

typedef struct {
    ts_tensor_shape input_shape;

    ts_tensor_shape pool_size;
    ts_layer_pooling_type type;
} _layer_pooling_2d_backend;

typedef struct {
    ts_tensor_shape kernel_size;

    // Shape is (kernel_size.w * kernel_size.h, in_filters, out_filters)
    ts_tensor* kernels;
    // Shape is out_shape
    ts_tensor* biases; 

    ts_u32 stride_x;
    ts_u32 stride_y;

    ts_tensor_shape input_shape;
    ts_tensor_shape padded_shape;

    // Training mode
    ts_param_change kernels_change;
    ts_param_change biases_change;
} _layer_conv_2d_backend;

typedef struct ts_layer {
    // Initialized in layer_create
    ts_layer_type type;
    ts_b32 training_mode;

    // Should be set by layer in create function
    ts_tensor_shape shape;

    union {
        _layer_reshape_backend reshape_backend;
        _layer_dense_backend dense_backend;
        _layer_activation_backend activation_backend;
        _layer_dropout_backend dropout_backend;
        _layer_flatten_backend flatten_backend;
        _layer_pooling_2d_backend pooling_2d_backend;
        _layer_conv_2d_backend conv_2d_backend;
    };
} ts_layer;

void ts_param_init(ts_tensor* param, ts_param_init_type input_type, ts_u64 in_size, ts_u64 out_size);

// TODO: consistent underscoring for private stuff
typedef struct ts_layers_cache_node {
    ts_tensor* t;
    struct ts_layers_cache_node* next;
} ts_layers_cache_node;

typedef struct ts_layers_cache {
    mg_arena* arena;

    ts_layers_cache_node* first;
    ts_layers_cache_node* last;
} ts_layers_cache;

void ts_layers_cache_push(ts_layers_cache* cache, ts_tensor* t);
ts_tensor* ts_layers_cache_pop(ts_layers_cache* cache);

typedef void (_layer_create_func)(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
typedef void (_layer_feedforward_func)(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
typedef void (_layer_backprop_func)(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
typedef void (_layer_apply_changes_func)(ts_layer* l, const ts_optimizer* optim);
typedef void (_layer_delete_func)(ts_layer* l);
typedef void (_layer_save_func)(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index);
typedef void (_layer_load_func)(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

typedef struct {
    _layer_create_func* create;
    _layer_feedforward_func* feedforward;
    _layer_backprop_func* backprop;
    _layer_apply_changes_func* apply_changes;
    _layer_delete_func* delete;
    _layer_save_func* save;
    _layer_load_func* load;
} _layer_func_defs;

// These functions are implemented in specific layers_*.c files

void _layer_null_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
void _layer_null_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache);
void _layer_null_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
void _layer_null_apply_changes(ts_layer* l, const ts_optimizer* optim);
void _layer_null_delete(ts_layer* l);
void _layer_null_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index);
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
void _layer_dense_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index);
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
void _layer_conv_2d_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index);
void _layer_conv_2d_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

#endif // LAYERS_INTERNAL_H

