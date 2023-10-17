#include "optimizers.h"

#include <stdio.h>

void param_change_create(mg_arena* arena, param_change* out, tensor_shape shape) {
    out->change = tensor_create(arena, shape);
    out->_V = tensor_create(arena, shape);
    out->_S = tensor_create(arena, shape);
}

typedef void(_param_update_func)(const optimizer*, tensor*, param_change*);

void _null_param_update(const optimizer* optim, tensor* param, param_change* change);
void _sgd_param_update(const optimizer* optim, tensor* param, param_change* change);
void _rms_prop_param_update(const optimizer* optim, tensor* param, param_change* change);
void _adam_param_update(const optimizer* optim, tensor* param, param_change* change);

static _param_update_func* _update_funcs[OPTIMIZER_COUNT] = {
    [OPTIMIZER_NULL] = _null_param_update,
    [OPTIMIZER_SGD] = _sgd_param_update,
    [OPTIMIZER_RMS_PROP] = _rms_prop_param_update,
    [OPTIMIZER_ADAM] = _adam_param_update,
};

void param_change_update(const optimizer* optim, tensor* param, param_change* change) {
    if (optim->type >= OPTIMIZER_COUNT) {
        fprintf(stderr, "Cannot update param: Invalid optimizer type\n");
        return;
    }

    _update_funcs[optim->type](optim, param, change);
}

void _null_param_update(const optimizer* optim, tensor* param, param_change* change) {
    UNUSED(optim);
    UNUSED(param);
    UNUSED(change);
}
void _sgd_param_update(const optimizer* optim, tensor* param, param_change* change) {
    f32 beta = optim->sgd.momentum;

    // Averaging change over batch 
    tensor_scale_ip(change->change, change->change, 1.0f / (f32)optim->_batch_size);

    // V_t = beta * V_t-1 + (1 - beta) * d
    tensor_scale_ip(change->_V, change->_V, beta);
    tensor_scale_ip(change->change, change->change, 1.0f - beta);
    tensor_add_ip(change->_V, change->_V, change->change);

    // param = param - (learning_rate * V)
    tensor_scale_ip(change->change, change->_V, optim->learning_rate);
    tensor_sub_ip(param, param, change->change);
}
void _rms_prop_param_update(const optimizer* optim, tensor* param, param_change* change) {
    f32 beta = optim->rms_prop.beta;
    f32 epsilon = optim->rms_prop.epsilon;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Averaging change over batch 
    tensor_scale_ip(change->change, change->change, 1.0f / (f32)optim->_batch_size);
    tensor* real_change = tensor_copy(scratch.arena, change->change, false);

    // S_t = beta * S_t-1 + (1 - beta) * d^2
    tensor_scale_ip(change->_S, change->_S, beta);
    tensor_component_mul_ip(change->change, change->change, change->change);
    tensor_scale_ip(change->change, change->change, 1.0f - beta);
    tensor_add_ip(change->_S, change->_S, change->change);

    // param = param - (learning_rate / sqrt(S + epsilon)) * dW
    tensor* sqrt_S = tensor_copy(scratch.arena, change->_S, false);
    u64 size = (u64)sqrt_S->shape.width * sqrt_S->shape.height * sqrt_S->shape.depth;
    for (u64 i = 0; i < size; i++) {
        sqrt_S->data[i] += epsilon;
    }
    tensor_sqrt_ip(sqrt_S, sqrt_S);

    tensor_component_div_ip(real_change, real_change, sqrt_S);
    tensor_scale_ip(real_change, real_change, optim->learning_rate);

    tensor_sub_ip(param, param, real_change);

    mga_scratch_release(scratch);
}
void _adam_param_update(const optimizer* optim, tensor* param, param_change* change) {
}
