/**
 * @file optimizers.h
 * @brief Parameter optimizers for the neural networks
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "base_defs.h"
#include "os.h"
#include "tensor.h"

/// Type of optimizers
typedef enum {
    /// Does nothing
    TS_OPTIMIZER_NULL = 0,

    /// Stochastic Gradient Descent
    TS_OPTIMIZER_SGD,

    /// Root Mean Square Propagation
    TS_OPTIMIZER_RMS_PROP,

    /// Adaptive Moment Estimation
    TS_OPTIMIZER_ADAM,

    /// Number of optimizers
    TS_OPTIMIZER_COUNT
} ts_optimizer_type;

/// Stochastic Gradient Descent Parameters
typedef struct {
    /**
     * @brief Exponentially moving average param
     * 
     * Typically `0.9` <br>
     * `0.0f <= momentum <= 1.0f` <br>
     * `V = momentum * V_prev + (1 - momentum) * dW`
     */ 
    ts_f32 momentum;
} ts_optimizer_sgd;

/// Root Mean Squared Propagation Parameters
typedef struct {
    /** 
     * @brief Discounting factor for old gradients
     *
     * Typically `0.999` <br>
     * `0.0f <= beta <= 1.0f` <br>
     * `S = beta * S_prev + (1 - beta) * (dW)^2` 
     */
    ts_f32 beta;
    
    /**
     * @brief For numerical stability
     *
     * `W = W - learning_rate * (dW / sqrt(S + epsilon))`
     */
    ts_f32 epsilon;
} ts_optimizer_rms_prop;

/// Adaptive Moment Estimation Parameters
typedef struct {
    /**
     * @brief Exponentially moving average param
     *
     * See `ts_optimizer_sgd` `momentum``
     */ 
    ts_f32 beta1;

    /**
     * @brief Discounting factor for old gradients
     *
     * See `ts_optimizer_rms_prop` `beta`
     */
    ts_f32 beta2;

    /**
     * @brief For numerical stability
     *
     * See `ts_optimizer_rms_prop` `epsilon`
     */
    ts_f32 epsilon;
} ts_optimizer_adam;

/// Full optimizer params
typedef struct {
    /**
     * @brief Scaling factor for changes
     * 
     * `W = W - learning_rate * (changes)`
     */
    ts_f32 learning_rate;

    /// Type of optimizer. Used for accessing the union
    ts_optimizer_type type;

    union {
        /// SGD params
        ts_optimizer_sgd sgd;
        /// RMS Prop params
        ts_optimizer_rms_prop rms_prop;
        /// Adam params
        ts_optimizer_adam adam;
    };

    /**
     * @brief Batch size during learning
     *
     * Does not need to be set in network_train_desc (will be set by neural network)
     */
    ts_u32 _batch_size;
} ts_optimizer;

/**
 * @brief Storage for changes in trainable parameters
 *
 * Do not modify any members of a `ts_param_change` directly. <br>
 * If a layer has trainable params, the changes should
 * be accumulated and applied with a `ts_param_change`
 */
typedef struct {
    /// Mutex for changing the other params
    ts_mutex* _mutex;

    /**
     * @brief Change in param
     * 
     * Should not be updated directly
     */
    ts_tensor* _change;

    /// State for SGD and Adam
    ts_tensor* _V;

    /// State for RMS Prop ans Adam
    ts_tensor* _S;
} ts_param_change;

/**
 * @brief Initializes a `ts_param_change` in `out`
 *
 * This does not return a param_change because layers should already have a `ts_param_change` member. <br>
 * See `layers_dense.c` for an example
 *
 * @param arena Arena for param_change
 * @param out Output of creation
 * @param shape Shape of param
 */
void ts_param_change_create(mg_arena* arena, ts_param_change* out, ts_tensor_shape shape);
/**
 * @brief Adds `addend` to `param_change`
 *
 * Layers must use this function for thread safety
 */
void ts_param_change_add(ts_param_change* param_change, ts_tensor* addend);
/**
 * @brief Applies any changes in `param_change` to `param`
 *
 * @param optim Optimizer to use for updating
 * @param param Parameter to update
 * @param param_change Param change for `param`
 */
void ts_param_change_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change);
/**
 * @brief Deletes `param_change`
 *
 * This is necessary because of the mutex
 */
void ts_param_change_delete(ts_param_change* param_change);

#endif // OPTIMIZERS_H

