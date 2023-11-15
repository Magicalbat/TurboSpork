#ifndef COST_H
#define COST_H

#include "base/base.h"
#include "tensor/tensor.h"

typedef enum {
    COST_NULL = 0,

    COST_MEAN_SQUARED_ERROR,

    COST_CATEGORICAL_CROSS_ENTROPY,

    COST_COUNT
} cost_type;

f32 cost_func(cost_type type, const tensor* in, const tensor* desired_out);
void cost_grad(cost_type type, tensor* in_out, const tensor* desired_out);

#endif // COST_H
