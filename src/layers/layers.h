#ifndef LAYERS_H
#define LAYERS_H

#include "mg/mg_arena.h"

#include "base/base.h"
#include "tensor/tensor.h"
#include "optimizers/optimizers.h"

typedef enum {
    LAYER_NULL = 0,
    LAYER_INPUT,
    LAYER_DENSE,
    LAYER_ACTIVATION,
    LAYER_DROPOUT,
    LAYER_FLATTEN,
    LAYER_POOLING,

    LAYER_COUNT
} layer_type;

typedef enum {
    ACTIVATION_NULL = 0,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_SOFTMAX,

    ACTIVATION_COUNT
} layer_activation_type;

typedef enum {
    POOLING_NULL = 0,
    POOLING_MAX,
    POOLING_AVG,
    POOLING_L2,

    POOLING_COUNT
} layer_pooling_type;

typedef struct {
    tensor_shape shape;
} layer_input_desc;

typedef struct {
    u32 size;

    // TODO: weight initialization options
} layer_dense_desc;

typedef struct {
    layer_activation_type type;
} layer_activation_desc;

typedef struct {
    f32 keep_rate;
} layer_dropout_desc;

typedef struct {
    // Only supports 2d right now
    tensor_shape pool_size;

    layer_pooling_type type;
} layer_pooling_desc;

typedef struct {
    layer_type type;
    b32 training_mode;

    union {
        layer_input_desc input;
        layer_dense_desc dense;
        layer_activation_desc activation;
        layer_dropout_desc dropout;
        layer_pooling_desc pooling;
    };
} layer_desc;

// Defined in layers_internal.h
typedef struct layer layer;
typedef struct layers_cache layers_cache;

string8 layer_get_name(layer_type type);
layer_type layer_from_name(string8 name);

layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape);
// cache can be NULL, only used for training
void layer_feedforward(layer* l, tensor* in_out, layers_cache* cache); 
void layer_backprop(layer* l, tensor* delta, layers_cache* cache);
void layer_apply_changes(layer* l, const optimizer* optim);
void layer_delete(layer* l);
// Saves layer params, not anything that would be in the desc
void layer_save(mg_arena* arena, tensor_list* list, layer* l, u32 index);
// Loads layer params, not anything that would be in the desc
// Layer needs to be created with the correct type
void layer_load(layer* l, const tensor_list* list, u32 index);

void layer_desc_save(mg_arena* arena, string8_list* list, const layer_desc* desc);
layer_desc layer_desc_load(string8 str);

#endif // LAYERS_H

