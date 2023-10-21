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
    layer_type type;
    b32 training_mode;

    union {
        layer_input_desc input;
        layer_dense_desc dense;
        layer_activation_desc activation;
    };
} layer_desc;

typedef struct {
    tensor* weight;
    tensor* bias;

    // Training mode
    param_change weight_change;
    param_change bias_change;
} layer_dense_backend;

typedef struct {
    layer_activation_type type;
} layer_activation_backend;

typedef struct {
    // Initialized in layer_create
    layer_type type;
    b32 training_mode;

    // Training mode
    tensor* prev_input;

    // Should be set by layer
    tensor_shape shape;

    union {
        layer_dense_backend dense_backend;
        layer_activation_backend activation_backend;
    };
} layer;

string8 layer_get_name(layer_type type);
layer_type layer_from_name(string8 name);

layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape);
void layer_feedforward(layer* l, tensor* in_out); 
void layer_backprop(layer* l, tensor* delta);
void layer_apply_changes(layer* l, const optimizer* optim);

#endif // LAYERS_H

