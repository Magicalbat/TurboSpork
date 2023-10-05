#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

typedef enum {
    OPTIMIZER_NULL = 0,

    // Stochastic Gradient Descent
    OPTIMIZER_SGD,

    // TODO: RSM Prop?

    // Adaptive Moment Estimation
    OPTIMIZER_ADAM,
} optimizer_type;

#endif // OPTIMIZERS_H
