#include "costs.h"

#include <stdio.h>
#include <math.h>

typedef f32 (_cost_func)(const tensor*, const tensor*);
typedef void (_cost_deriv)(tensor*, const tensor*);

static f32 null_func(const tensor* in, const tensor* desired_out);
static void null_deriv(tensor* in_out, const tensor* desired_out);
static f32 quadratic_func(const tensor* in, const tensor* desired_out);
static void quadratic_deriv(tensor* in_out, const tensor* desired_out);
static f32 cross_entropy_func(const tensor* in, const tensor* desired_out);
static void cross_entropy_deriv(tensor* in_out, const tensor* desired_out);

static _cost_func* _funcs[COST_COUNT] = {
    [COST_NULL] = null_func,
    [COST_QUADRATIC] = quadratic_func,
    [COST_CROSS_ENTROPY] = cross_entropy_func
};
static _cost_deriv* _derivs[COST_COUNT] = {
    [COST_NULL] = null_deriv,
    [COST_QUADRATIC] = quadratic_deriv,
    [COST_CROSS_ENTROPY] = cross_entropy_deriv
};

f32 cost_func(cost_type type, const tensor* in, const tensor* desired_out) {
    if (type >= COST_COUNT) {
        fprintf(stderr, "Invalid cost function\n");
        return 0.0f;
    }

    if (!tensor_shape_eq(in->shape, desired_out->shape)) {
        fprintf(stderr, "Invalid input to cost function: shapes must align\n");
        return 0.0f;
    }

    return _funcs[type](in, desired_out);
}
void cost_deriv(cost_type type, tensor* in_out, const tensor* desired_out) {
    if (type >= COST_COUNT) { fprintf(stderr, "Invalid cost derivative\n");
        return;
    }
    if (!tensor_shape_eq(in_out->shape, desired_out->shape)) {
        fprintf(stderr, "Invalid input to cost derivative: shapes must align\n");
        return;
    }

    _derivs[type](in_out, desired_out);
}

static f32 null_func(const tensor* in, const tensor* desired_out) {
    UNUSED(in);
    UNUSED(desired_out);

    return 0.0f;
}
static void null_deriv(tensor* in_out, const tensor* desired_out) {
    UNUSED(in_out);
    UNUSED(desired_out);
}
static f32 quadratic_func(const tensor* in, const tensor* desired_out) {
    f32 sum = 0.0f;

    u64 size = (u64)in->shape.width * in->shape.height * in->shape.depth;
    for (u64 i = 0; i < size; i++) {
        sum += 0.5f * (in->data[i] - desired_out->data[i]) * (in->data[i] - desired_out->data[i]);
    }

    // TODO: average or sum?
    return sum;
}
static void quadratic_deriv(tensor* in_out, const tensor* desired_out) {
    u64 size = (u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;
    for (u64 i = 0; i < size; i++) {
        in_out->data[i] = (in_out->data[i] - desired_out->data[i]);
    }
}

// TODO: figure out what the cross entropy function really is
static f32 cross_entropy_func(const tensor* in, const tensor* desired_out) {
    UNUSED(in);
    UNUSED(desired_out);

    return 0.0f;
}
static void cross_entropy_deriv(tensor* in_out, const tensor* desired_out) {
    UNUSED(in_out);
    UNUSED(desired_out);
}
