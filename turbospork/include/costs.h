/**
 * @file costs.h
 * @brief Costs of neural networks
 */

#ifndef TS_COST_H
#define TS_COST_H

#include "base_defs.h"
#include "tensor.h"

/**
 * @brief Cost types
 */
typedef enum {
    /**
     * @brief Does not perform any operation on the inputs
     */
    TS_COST_NULL = 0,

    /**
     * @brief Computes the mean squared error
     * 
     * C(a, y) = 0.5(a - y)^2 <br>
     * C'(a, y) = a - y
     */
    TS_COST_MEAN_SQUARED_ERROR,

    /**
     * @brief Computes the categorical cross entorpy error
     * 
     * C(a, y) = y * ln(a) <br>
     * C'(a, y) = -y / a
     */
    TS_COST_CATEGORICAL_CROSS_ENTROPY,

    /// Number for costs
    TS_COST_COUNT
} ts_cost_type;

/**
 * @brief Computes the cost
 *
 * `in` and `desired_out` must be the same shape
 *
 * @param type Which cost function to use
 * @param in Input of cost function (typically the neural network output)
 * @param desired_out True value of input (typically from some training data)
 */
ts_f32 ts_cost_func(ts_cost_type type, const ts_tensor* in, const ts_tensor* desired_out);
/**
 * @brief Computes the gradient of the cost function
 * 
 * `in_out` and `desired_out` must be the same shape
 *
 * @param type Which cost function to use
 * @param in_out Input to cost gradient and where the output will be stored
 * @param desired_out True value of input (typically from some training data)
 */
void ts_cost_grad(ts_cost_type type, ts_tensor* in_out, const ts_tensor* desired_out);

#endif // TS_COST_H
