#ifndef COST_H
#define COST_H

#include "base_defs.h"
#include "tensor.h"

typedef enum {
    TS_COST_NULL = 0,

    TS_COST_MEAN_SQUARED_ERROR,

    TS_COST_CATEGORICAL_CROSS_ENTROPY,

    TS_COST_COUNT
} ts_cost_type;

ts_f32 ts_cost_func(ts_cost_type type, const ts_tensor* in, const ts_tensor* desired_out);
void ts_cost_grad(ts_cost_type type, ts_tensor* in_out, const ts_tensor* desired_out);

#endif // COST_H
