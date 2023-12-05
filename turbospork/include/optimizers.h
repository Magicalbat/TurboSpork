#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base_defs.h"
#include "os.h"
#include "tensor.h"

typedef enum {
    TS_OPTIMIZER_NULL = 0,

    // Stochastic Gradient Descent
    TS_OPTIMIZER_SGD,

    // Root Mean Square Propagation
    TS_OPTIMIZER_RMS_PROP,

    // Adaptive Moment Estimation
    TS_OPTIMIZER_ADAM,

    TS_OPTIMIZER_COUNT
} ts_optimizer_type;

typedef struct {
    // Float 0.0f <= momentum <= 1.0f
    // Exponentially moving average param
    // V = momentum * V_prev + (1 - momentum) * dW 
    // Typically 0.9
    ts_f32 momentum;
} ts_optimizer_sgd;

typedef struct {
    // Float 0.0f <= beta <= 1.0f
    // Discounting factor for old gradients
    // S = beta * S_prev + (1 - beta) * (dW)^2
    // W = W - learning_rate * (dW / sqrt(S + epsilon))
    // Typically 0.999
    ts_f32 beta;
    
    // Parameter for numerical stability
    ts_f32 epsilon;
} ts_optimizer_rms_prop;

typedef struct {
    // Exponentially moving average param
    // See SGD
    ts_f32 beta1;

    // RMS Prop params
    // See RMS prop
    ts_f32 beta2;
    ts_f32 epsilon;
} ts_optimizer_adam;

typedef struct {
    // Scaling factor for changes
    // W = W - learning_rate * (changes)
    ts_f32 learning_rate;

    ts_optimizer_type type;

    union {
        ts_optimizer_sgd sgd;
        ts_optimizer_rms_prop rms_prop;
        ts_optimizer_adam adam;
    };

    // Does not need to be set in network_train_desc
    ts_u32 _batch_size;
} ts_optimizer;

// This is for layers
// If a ts_tensor is updated through training,
// the change should be stored in one of these
typedef struct {
    ts_mutex* mutex;

    // Change should be set by layers before calling param_change_update
    ts_tensor* change;

    // These two are "private" and should not be modified by layres

    // For SGD and Adam
    ts_tensor* _V;

    // For RMS Prop ans Adam
    ts_tensor* _S;
} ts_param_change;

void ts_param_change_create(mg_arena* arena, ts_param_change* out, ts_tensor_shape shape);
// Adds addend to change->change
void ts_param_change_add(ts_param_change* change, ts_tensor* addend);
// Updates param, and fills change->change with zeros
void ts_param_change_update(const ts_optimizer* optim, ts_tensor* param, ts_param_change* change);
void ts_param_change_delete(ts_param_change* change);

#endif // OPTIMIZERS_H

