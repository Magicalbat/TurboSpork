#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base/base.h"

typedef enum {
    OPTIMIZER_NULL = 0,

    // Stochastic Gradient Descent
    OPTIMIZER_SGD,

    // TODO: RSM Prop?

    // Adaptive Moment Estimation
    OPTIMIZER_ADAM,
} optimizer_type;

typedef struct {
    f32 momentum;
} optimizer_sgd_desc;

typedef struct {
    f32 beta1;
    f32 beta2;
} optimizer_adam_desc;

typedef struct {
    optimizer_type type;

    union {
        optimizer_sgd_desc sgd;
        optimizer_adam_desc adam;
    };
} optimizer_desc;

#endif // OPTIMIZERS_H

