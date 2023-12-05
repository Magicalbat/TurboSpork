#ifndef LAYERS_H
#define LAYERS_H

#include "mg/mg_arena.h"

#include "base_defs.h"
#include "str.h"
#include "tensor.h"
#include "optimizers.h"

typedef enum {
    TS_LAYER_NULL = 0,
    TS_LAYER_INPUT,
    TS_LAYER_RESHAPE,
    TS_LAYER_DENSE,
    TS_LAYER_ACTIVATION,
    TS_LAYER_DROPOUT,
    TS_LAYER_FLATTEN,
    TS_LAYER_POOLING_2D,
    TS_LAYER_CONV_2D,

    TS_LAYER_COUNT
} ts_layer_type;

typedef enum {
    TS_PARAM_INIT_NULL = 0,

    // Fills param with zeors
    TS_PARAM_INIT_ZEROS,
    // Fills param with zeors
    TS_PARAM_INIT_ONES,

    // Xavier Glorot uniform
    TS_PARAM_INIT_XAVIER_UNIFORM,
    // Xavier Glorot normal
    TS_PARAM_INIT_XAVIER_NORMAL,

    // He/Kaiming uniform
    TS_PARAM_INIT_HE_UNIFORM,
    // He/Kaiming normal
    TS_PARAM_INIT_HE_NORMAL,

    TS_PARAM_INIT_COUNT
} ts_param_init_type;

typedef enum {
    TS_ACTIVATION_NULL = 0,
    TS_ACTIVATION_SIGMOID,
    TS_ACTIVATION_TANH,
    TS_ACTIVATION_RELU,
    TS_ACTIVATION_LEAKY_RELU,
    TS_ACTIVATION_SOFTMAX,

    TS_ACTIVATION_COUNT
} ts_layer_activation_type;

typedef enum {
    TS_POOLING_NULL = 0,
    TS_POOLING_MAX,
    TS_POOLING_AVG,

    TS_POOLING_COUNT
} ts_layer_pooling_type;

typedef struct {
    ts_tensor_shape shape;
} ts_layer_input_desc;

typedef struct {
    ts_tensor_shape shape;
} ts_layer_reshape_desc;

typedef struct {
    ts_u32 size;

    // Default of PARAM_INIT_ZEROS
    ts_param_init_type bias_init;
    // Default of PARAM_INIT_XAVIER_UNIFORM
    ts_param_init_type weight_init;
} ts_layer_dense_desc;

typedef struct {
    // Default of ACTIVATION_RELU
    ts_layer_activation_type type;
} ts_layer_activation_desc;

typedef struct {
    ts_f32 keep_rate;
} ts_layer_dropout_desc;

typedef struct {
    ts_tensor_shape pool_size;

    // Default of POOLING_MAX
    ts_layer_pooling_type type;
} ts_layer_pooling_2d_desc;

typedef struct {
    ts_u32 num_filters;

    // Only 2d
    ts_tensor_shape kernel_size;

    // Adds padding to input
    // If strides are 1, and padding is true,
    // then the output size is the same as the input size
    ts_b32 padding;

    // Strides for filter
    // Defaults to 1
    ts_u32 stride_x;
    ts_u32 stride_y;

    // Default of PARAM_INIT_HE_NORMAL
    ts_param_init_type kernels_init;
    // Default of PARAM_INIT_ZEROS
    ts_param_init_type biases_init;
} ts_layer_conv_2d_desc;

typedef struct {
    ts_layer_type type;
    ts_b32 training_mode;

    union {
        ts_layer_input_desc input;
        ts_layer_reshape_desc reshape;
        ts_layer_dense_desc dense;
        ts_layer_activation_desc activation;
        ts_layer_dropout_desc dropout;
        ts_layer_pooling_2d_desc pooling_2d;
        ts_layer_conv_2d_desc conv_2d;
    };
} ts_layer_desc;

// Defined in layers_internal.h
typedef struct ts_layer ts_layer;
typedef struct ts_layers_cache ts_layers_cache;

ts_string8 ts_layer_get_name(ts_layer_type type);
ts_layer_type ts_layer_from_name(ts_string8 name);

ts_layer* ts_layer_create(mg_arena* arena, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
// cache can be NULL, only used for training
void ts_layer_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache); 
void ts_layer_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
void ts_layer_apply_changes(ts_layer* l, const ts_optimizer* optim);
void ts_layer_delete(ts_layer* l);
// Saves layer params, not anything that would be in the desc
void ts_layer_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index);
// Loads layer params, not anything that would be in the desc
// Layer needs to be created with the correct type
void ts_layer_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

ts_layer_desc ts_layer_desc_default(ts_layer_type type);
ts_layer_desc ts_layer_desc_apply_default(const ts_layer_desc* desc);

void ts_layer_desc_save(mg_arena* arena, ts_string8_list* list, const ts_layer_desc* desc);
ts_layer_desc ts_layer_desc_load(ts_string8 str);

#endif // LAYERS_H

