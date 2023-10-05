#ifndef LAYERS_H
#define LAYERS_H

#include "base/base.h"
#include "tensor/tensor.h"

typedef enum {
    LAYER_NULL = 0,
    LAYER_DENSE,
    LAYER_ACTIVATION,
} layer_type;

typedef struct {
    // Shape info?
    // FOrward pass
    // Backward pass
} layer_backend;

typedef struct {
    layer_type type;
} layer;

#endif // LAYERS_H
