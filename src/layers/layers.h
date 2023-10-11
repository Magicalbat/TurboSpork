#ifndef LAYERS_H
#define LAYERS_H

#include "mg/mg_arena.h"

#include "base/base.h"
#include "tensor/tensor.h"

typedef enum {
    LAYER_NULL = 0,
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

    ACTIVATION_COUNT
} layer_activation_type;

typedef struct {
    u32 in_size;
    u32 out_size;

    // TODO: weight initialization options
} layer_dense_desc;

typedef struct {
    u32 size;
    layer_activation_type type;
} layer_activation_desc;

typedef struct {
    layer_type type;
    b32 training_mode;

    union {
        layer_dense_desc dense;
        layer_activation_desc activation;
    };
} layer_desc;

typedef struct {
    tensor* weight;
    tensor* bias;

    tensor* weight_change;
    tensor* bias_change;

    // for backprop
    tensor* prev_input;
} layer_dense_backend;

typedef struct {
    layer_activation_type func;
} layer_activation_backend;

typedef struct {
    layer_type type;
    b32 training_mode;

    tensor_shape input_shape;
    tensor_shape output_shape;

    union {
        layer_dense_backend dense_backend;
        layer_activation_backend activation_backend;
    };
} layer;

layer* layer_create(mg_arena* arena, const layer_desc* desc);
void layer_feedforward(layer* l, tensor* in_out); 
// TODO: Include previous input in backprop?
void layer_backprop(layer* l, tensor* delta);
void layer_apply_changes(layer* l);

#endif // LAYERS_H

