#ifndef LAYERS_H
#define LAYERS_H

#include "base/base.h"
#include "tensor/tensor.h"

typedef enum {
    LAYER_NULL = 0,
    LAYER_DENSE,
    LAYER_ACTIVATION,
} layer_type;

typedef enum {
    ACTIVATION_NULL = 0,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU,
    ACTIVATION_LEAKY_RELU,
} layer_activation_type;

typedef struct {
    u32 in_size;
    u32 out_size;
} layer_dense_desc;

typedef struct {
    layer_activation_type type;
} layer_activation_desc;

typedef struct {
    layer_type type;
    union {
        layer_dense_desc dense;
        layer_activation_desc activation;
    };
} layer_desc;



#endif // LAYERS_H

