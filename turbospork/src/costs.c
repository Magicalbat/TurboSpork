#include "costs.h"

#include <stdio.h>
#include <math.h>

typedef ts_f32 (_cost_func)(const ts_tensor*, const ts_tensor*);
typedef void (_cost_grad)(ts_tensor*, const ts_tensor*);

typedef struct {
    _cost_func* func;
    _cost_grad* grad;
} _cost;

static ts_f32 null_func(const ts_tensor* in, const ts_tensor* desired_out);
static void null_grad(ts_tensor* in_out, const ts_tensor* desired_out);
static ts_f32 mean_squared_func(const ts_tensor* in, const ts_tensor* desired_out);
static void mean_squared_grad(ts_tensor* in_out, const ts_tensor* desired_out);
static ts_f32 cce_func(const ts_tensor* in, const ts_tensor* desired_out);
static void cce_grad(ts_tensor* in_out, const ts_tensor* desired_out);

static _cost _costs[TS_COST_COUNT] = {
    [TS_COST_NULL] = { null_func, null_grad },
    [TS_COST_MEAN_SQUARED_ERROR] = { mean_squared_func, mean_squared_grad },
    [TS_COST_CATEGORICAL_CROSS_ENTROPY] = { cce_func, cce_grad },
};

ts_f32 ts_cost_func(ts_cost_type type, const ts_tensor* in, const ts_tensor* desired_out) {
    if (type >= TS_COST_COUNT) {
        fprintf(stderr, "Invalid cost function\n");
        return 0.0f;
    }

    if (!ts_tensor_shape_eq(in->shape, desired_out->shape)) {
        fprintf(stderr, "Invalid input to cost function: shapes must align\n");
        return 0.0f;
    }

    return _costs[type].func(in, desired_out);
}
void ts_cost_grad(ts_cost_type type, ts_tensor* in_out, const ts_tensor* desired_out) {
    if (type >= TS_COST_COUNT) { fprintf(stderr, "Invalid cost gradient\n");
        return;
    }
    if (!ts_tensor_shape_eq(in_out->shape, desired_out->shape)) {
        fprintf(stderr, "Invalid input to cost gradient: shapes must align\n");
        return;
    }

    _costs[type].grad(in_out, desired_out);
}

static ts_f32 null_func(const ts_tensor* in, const ts_tensor* desired_out) {
    TS_UNUSED(in);
    TS_UNUSED(desired_out);

    return 0.0f;
}
static void null_grad(ts_tensor* in_out, const ts_tensor* desired_out) {
    TS_UNUSED(in_out);
    TS_UNUSED(desired_out);
}
static ts_f32 mean_squared_func(const ts_tensor* in, const ts_tensor* desired_out) {
    ts_f32 sum = 0.0f;

    ts_u64 size = (ts_u64)in->shape.width * in->shape.height * in->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        sum += 0.5f * (in->data[i] - desired_out->data[i]) * (in->data[i] - desired_out->data[i]);
    }

    return sum / (ts_f32)size;
}
static void mean_squared_grad(ts_tensor* in_out, const ts_tensor* desired_out) {
    ts_u64 size = (ts_u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        in_out->data[i] = (in_out->data[i] - desired_out->data[i]);
    }
}

static ts_f32 cce_func(const ts_tensor* in, const ts_tensor* desired_out) {
    ts_f32 sum = 0.0f;

    ts_u64 size = (ts_u64)in->shape.width * in->shape.height * in->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        sum += desired_out->data[i] * logf(in->data[i]);
    }

    return -sum;
}
static void cce_grad(ts_tensor* in_out, const ts_tensor* desired_out) {
    ts_u64 size = (ts_u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        in_out->data[i] = -desired_out->data[i] / (in_out->data[i] + 1e-8);
    }
}

