#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base/base.h"
#include "os/os.h"
#include "tensor/tensor.h"

typedef enum {
    OPTIMIZER_NULL = 0,

    // Stochastic Gradient Descent
    OPTIMIZER_SGD,

    // Root Mean Square Propagation
    OPTIMIZER_RMS_PROP,

    // Adaptive Moment Estimation
    OPTIMIZER_ADAM,

    OPTIMIZER_COUNT
} optimizer_type;

typedef struct {
    // Float 0.0f <= momentum <= 1.0f
    // Exponentially moving average param
    // V = momentum * V_prev + (1 - momentum) * dW 
    // Typically 0.9
    f32 momentum;
} optimizer_sgd;

typedef struct {
    // Float 0.0f <= beta <= 1.0f
    // Discounting factor for old gradients
    // S = beta * S_prev + (1 - beta) * (dW)^2
    // W = W - learning_rate * (dW / sqrt(S + epsilon))
    // Typically 0.999
    f32 beta;
    
    // Parameter for numerical stability
    f32 epsilon;
} optimizer_rms_prop;

typedef struct {
    // Exponentially moving average param
    // See SGD
    f32 beta1;

    // RMS Prop params
    // See RMS prop
    f32 beta2;
    f32 epsilon;
} optimizer_adam;

typedef struct {
    // Scaling factor for changes
    // W = W - learning_rate * (changes)
    f32 learning_rate;

    optimizer_type type;

    union {
        optimizer_sgd sgd;
        optimizer_rms_prop rms_prop;
        optimizer_adam adam;
    };

    // Does not need to be set in network_train_desc
    u32 _batch_size;
} optimizer;

// This is for layers
// If a tensor is updated through training,
// the change should be stored in one of these
typedef struct {
    os_thread_mutex* mutex;

    // Change should be set by layers before calling param_change_update
    tensor* change;

    // These two are "private" and should not be modified by layres

    // For SGD and Adam
    tensor* _V;

    // For RMS Prop ans Adam
    tensor* _S;
} param_change;

void param_change_create(mg_arena* arena, param_change* out, tensor_shape shape);
// Adds addend to change->change
void param_change_add(param_change* change, tensor* addend);
// Updates param, and fills change->change with zeros
void param_change_update(const optimizer* optim, tensor* param, param_change* change);
void param_change_delete(param_change* change);

#endif // OPTIMIZERS_H

