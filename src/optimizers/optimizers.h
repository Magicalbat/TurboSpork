#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base/base.h"

typedef enum {
    OPTIMIZER_NULL = 0,

    // Stochastic Gradient Descent
    OPTIMIZER_SGD,

    // Root Mean Square Propagation
    OPTIMIZER_RMS_PROP,

    // Adaptive Moment Estimation
    OPTIMIZER_ADAM,
} optimizer_type;

typedef struct {
    // Float 0.0f <= momentum <= 1.0f
    // Exponentially moving average param
    // V = momentum * V_prev + (1 - momentum) * dW 
    // Typically 0.9
    f32 momentum;
} optimizer_sgd_desc;

typedef struct {
    // Float 0.0f <= beta <= 1.0f
    // Discounting factor for old gradients
    // S = beta * S_prev + (1 - beta) * (dW)^2
    // W = W - learning_rate * (dW / sqrt(S + epsilon))
    // Typically 0.999
    f32 beta;
    
    // Parameter for numerical stability
    f32 epsilon;
} optimizer_rms_prop_desc;

typedef struct {
    // Exponentially moving average param
    // See SGD
    f32 beta1;

    // RMS Prop params
    // See RMS prop
    f32 beta2;
    f32 epsilon;
} optimizer_adam_desc;

typedef struct {
    // Scaling factor for changes
    // W = W - learning_rate * (changes)
    f32 learning_rate;

    optimizer_type type;

    union {
        optimizer_sgd_desc sgd;
        optimizer_rms_prop_desc rms_prop;
        optimizer_adam_desc adam;
    };
} optimizer_desc;

#endif // OPTIMIZERS_H

