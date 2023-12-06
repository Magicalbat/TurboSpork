#include "optimizers.h"

#include <stdio.h>
#include <math.h>

void ts_param_change_create(mg_arena* arena, ts_param_change* out, ts_tensor_shape shape) {
    out->_mutex = ts_mutex_create(arena);

    out->_change = ts_tensor_create(arena, shape);
    out->_V = ts_tensor_create(arena, shape);
    out->_S = ts_tensor_create(arena, shape);
}
void ts_param_change_add(ts_param_change* param_change, ts_tensor* addend) {
    ts_mutex_lock(param_change->_mutex);

    ts_tensor_add_ip(param_change->_change, param_change->_change, addend);

    ts_mutex_unlock(param_change->_mutex);
}
void ts_param_change_delete(ts_param_change* param_change) {
    ts_mutex_destroy(param_change->_mutex);
}

typedef void(_param_apply_func)(const ts_optimizer*, ts_tensor*, ts_param_change*);

void _null_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change);
void _sgd_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change);
void _rms_prop_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change);
void _adam_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change);

static _param_apply_func* _apply_funcs[TS_OPTIMIZER_COUNT] = {
    [TS_OPTIMIZER_NULL] = _null_param_apply,
    [TS_OPTIMIZER_SGD] = _sgd_param_apply,
    [TS_OPTIMIZER_RMS_PROP] = _rms_prop_param_apply,
    [TS_OPTIMIZER_ADAM] = _adam_param_apply,
};

void ts_param_change_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change) {
    if (optim->type >= TS_OPTIMIZER_COUNT) {
        fprintf(stderr, "Cannot update param: Invalid optimizer type\n");
        return;
    }

    ts_mutex_lock(param_change->_mutex);

    _apply_funcs[optim->type](optim, param, param_change);
    ts_tensor_fill(param_change->_change, 0.0f);

    ts_mutex_unlock(param_change->_mutex);
}

void _null_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change) {
    TS_UNUSED(optim);
    TS_UNUSED(param);
    TS_UNUSED(param_change);
}
void _sgd_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change) {
    ts_f32 beta = optim->sgd.momentum;

    // Averaging change over batch 
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (ts_f32)optim->_batch_size);

    // V_t = beta * V_t-1 + (1 - beta) * d
    ts_tensor_scale_ip(param_change->_V, param_change->_V, beta);
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - beta);
    ts_tensor_add_ip(param_change->_V, param_change->_V, param_change->_change);

    // param = param - (learning_rate * V)
    ts_tensor_scale_ip(param_change->_change, param_change->_V, optim->learning_rate);
    ts_tensor_sub_ip(param, param, param_change->_change);
}
void _rms_prop_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change) {
    ts_f32 beta = optim->rms_prop.beta;
    ts_f32 epsilon = optim->rms_prop.epsilon;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Averaging change over batch 
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (ts_f32)optim->_batch_size);
    ts_tensor* real_change = ts_tensor_copy(scratch.arena, param_change->_change, false);

    if (ts_tensor_is_zero(param_change->_S)) {
        // S_0 = d^2
        ts_tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
        ts_tensor_copy_ip(param_change->_S, param_change->_change);

    } else {
        // S_t = beta * S_t-1 + (1 - beta) * d^2
        ts_tensor_scale_ip(param_change->_S, param_change->_S, beta);
        ts_tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
        ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - beta);
        ts_tensor_add_ip(param_change->_S, param_change->_S, param_change->_change);
    }

    // param = param - (learning_rate / sqrt(S + epsilon)) * dW
    ts_tensor* sqrt_S = ts_tensor_copy(scratch.arena, param_change->_S, false);

    ts_u64 size = (ts_u64)sqrt_S->shape.width * sqrt_S->shape.height * sqrt_S->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        sqrt_S->data[i] += epsilon;
    }
    ts_tensor_sqrt_ip(sqrt_S, sqrt_S);

    ts_tensor_component_div_ip(real_change, real_change, sqrt_S);
    ts_tensor_scale_ip(real_change, real_change, optim->learning_rate);

    ts_tensor_sub_ip(param, param, real_change);

    mga_scratch_release(scratch);
}
void _adam_param_apply(const ts_optimizer* optim, ts_tensor* param, ts_param_change* param_change) {
    ts_optimizer_adam adam = optim->adam;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Averaging change over batch 
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f / (ts_f32)optim->_batch_size);
    ts_tensor* real_change = ts_tensor_copy(scratch.arena, param_change->_change, false);

    // V_t = beta * V_t-1 + (1 - beta) * d
    ts_tensor_scale_ip(param_change->_V, param_change->_V, adam.beta1);
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - adam.beta1);
    ts_tensor_add_ip(param_change->_V, param_change->_V, param_change->_change);

    // Putting original change back in change->_change
    ts_tensor_copy_ip(param_change->_change, real_change);

    // S_t = beta * S_t-1 + (1 - beta) * d^2
    ts_tensor_scale_ip(param_change->_S, param_change->_S, adam.beta2);
    ts_tensor_component_mul_ip(param_change->_change, param_change->_change, param_change->_change);
    ts_tensor_scale_ip(param_change->_change, param_change->_change, 1.0f - adam.beta2);
    ts_tensor_add_ip(param_change->_S, param_change->_S, param_change->_change);

    // param = param - (learning_rate / sqrt(S + epsilon)) * V
    ts_tensor* sqrt_S = ts_tensor_copy(scratch.arena, param_change->_S, false);
    ts_u64 size = (ts_u64)sqrt_S->shape.width * sqrt_S->shape.height * sqrt_S->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        sqrt_S->data[i] += adam.epsilon;
    }
    ts_tensor_sqrt_ip(sqrt_S, sqrt_S);

    ts_tensor_copy_ip(real_change, param_change->_V);

    ts_tensor_component_div_ip(real_change, real_change, sqrt_S);
    ts_tensor_scale_ip(real_change, real_change, optim->learning_rate);

    ts_tensor_sub_ip(param, param, real_change);

    mga_scratch_release(scratch);
}
