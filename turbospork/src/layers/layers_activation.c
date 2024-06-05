#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>
#include <math.h>

typedef void (_activ_func)(ts_tensor*);
typedef void (_activ_grad)(ts_tensor*, ts_tensor*, ts_tensor*);

typedef struct {
    _activ_func* func;
    _activ_grad* grad;
    ts_b32 cache_in;
    ts_b32 cache_out;
} _activation;

static void _null_func(ts_tensor* t);
static void _null_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _linear_func(ts_tensor* t);
static void _linear_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _sigmoid_func(ts_tensor* t);
static void _sigmoid_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _tanh_func(ts_tensor* t);
static void _tanh_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _relu_func(ts_tensor* t);
static void _relu_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _leaky_relu_func(ts_tensor* t);
static void _leaky_relu_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);
static void _softmax_func(ts_tensor* t);
static void _softmax_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta);

static _activation _activations[TS_ACTIVATION_COUNT] = {
    [TS_ACTIVATION_NULL] = { _null_func, _null_grad, false, false },
    [TS_ACTIVATION_LINEAR] = { _linear_func, _linear_grad, false, false },
    [TS_ACTIVATION_SIGMOID] = { _sigmoid_func, _sigmoid_grad, false, true },
    [TS_ACTIVATION_TANH] = { _tanh_func, _tanh_grad, false, true },
    [TS_ACTIVATION_RELU] = { _relu_func, _relu_grad, true, false },
    [TS_ACTIVATION_LEAKY_RELU] = { _leaky_relu_func, _leaky_relu_grad, true, false },
    [TS_ACTIVATION_SOFTMAX] = { _softmax_func, _softmax_grad, false, true },
};

void _layer_activation_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    ts_layer_activation_backend* activ = &out->activation_backend;

    activ->type = desc->activation.type;

    if (activ->type >= TS_ACTIVATION_COUNT) {
        fprintf(stderr, "Invalid activation type\n");

        activ->type = TS_ACTIVATION_NULL;
    }

    out->shape = prev_shape;
}
void _layer_activation_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    ts_layer_activation_backend* activ = &l->activation_backend;

    ts_b32 use_cache = cache != NULL && l->training_mode;

    // Cache input
    if (use_cache && _activations[activ->type].cache_in) {
        ts_tensor* input = ts_tensor_copy(cache->arena, in_out, false);
        ts_layers_cache_push(cache, input);
    }

    // Run activation
    _activations[activ->type].func(in_out);

    // Cache output
    if (use_cache && _activations[activ->type].cache_out) {
        ts_tensor* output = ts_tensor_copy(cache->arena, in_out, false);
        ts_layers_cache_push(cache, output);
    }
}
void _layer_activation_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    ts_layer_activation_backend* activ = &l->activation_backend;

    ts_tensor* prev_input = NULL;
    ts_tensor* prev_output = NULL;

    if (_activations[activ->type].cache_out) {
        prev_output = ts_layers_cache_pop(cache);
    }
    if (_activations[activ->type].cache_in) {
        prev_input = ts_layers_cache_pop(cache);
    }

    _activations[activ->type].grad(prev_input, prev_output, delta);
}

static void _null_func(ts_tensor* t) {
    TS_UNUSED(t);
}
static void _null_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_in);
    TS_UNUSED(prev_out);
    TS_UNUSED(delta);
}

static void _linear_func(ts_tensor* t) {
    TS_UNUSED(t);

    // Output of linear function equals input
}
static void _linear_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_in);
    TS_UNUSED(prev_out);
    TS_UNUSED(delta);

    // delta * 1 = delta
    // No code is required
}

#if TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

#define _LOOP_T(t) ts_u64 _size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth; \
    for (ts_u64 i = 0; i < _size; i++) 

static void _sigmoid_func(ts_tensor* t) {
    ts_f32* data = (ts_f32*)t->data;

    _LOOP_T(t) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}
static void _sigmoid_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_in);

    ts_f32* prev_out_data = (ts_f32*)prev_out->data;

    _LOOP_T(prev_out) {
        ts_f32 x = prev_out_data[i];
        prev_out_data[i] = x * (1.0f - x);
    }

    ts_tensor_component_mul_ip(delta, delta, prev_out);
}

static void _tanh_func(ts_tensor* t) {
    ts_f32* data = (ts_f32*)t->data;

    _LOOP_T(t) {
        data[i] = tanh(data[i]);
    }
}
static void _tanh_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_in);

    ts_f32* prev_out_data = (ts_f32*)prev_out->data;

    _LOOP_T(prev_out) {
        ts_f32 x = prev_out_data[i];
        prev_out_data[i] = 1.0f - x * x;
    }

    ts_tensor_component_mul_ip(delta, delta, prev_out);
}

static void _relu_func(ts_tensor* t) {
    ts_f32* data = (ts_f32*)t->data;

    _LOOP_T(t) {
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
    }
}
static void _relu_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_out);

    ts_f32* prev_in_data = (ts_f32*)prev_in->data;

    _LOOP_T(prev_in) {
        prev_in_data[i] = (prev_in_data[i] > 0.0f);
    }

    ts_tensor_component_mul_ip(delta, delta, prev_in);
}

static void _leaky_relu_func(ts_tensor* t) {
    ts_f32* data = (ts_f32*)t->data;

    _LOOP_T(t) {
        data[i] = data[i] > 0.0f ? data[i] : data[i] * 0.01f;
    }
}
static void _leaky_relu_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_out);

    ts_f32* prev_in_data = (ts_f32*)prev_in->data;

    _LOOP_T(prev_in) {
        prev_in_data[i] = prev_in_data[i] > 0.0f ? 1.0f : 0.01f;
    }

    ts_tensor_component_mul_ip(delta, delta, prev_in);
}

// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative 
static void _softmax_func(ts_tensor* t) {
    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;

    ts_f32* data = (ts_f32*)t->data;

    // Computing max for stability
    ts_f32 max_num = data[0];
    for (ts_u64 i = 0; i < size; i++) {
        if (data[i] > max_num) {
            max_num = data[i];
        }
    }

    // Exponentiation and exponential sum
    ts_f32 exp_sum = 0.0f;
    for (ts_u64 i = 0; i < size; i++) {
        data[i] = expf(data[i] - max_num);
        exp_sum += data[i];
    }

    exp_sum = 1.0f / exp_sum;
    for (ts_u64 i = 0; i < size; i++) {
        data[i] *= exp_sum;
    }
}
static void _softmax_grad(ts_tensor* prev_in, ts_tensor* prev_out, ts_tensor* delta) {
    TS_UNUSED(prev_in);

    ts_f32* prev_out_data = (ts_f32*)prev_out->data;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_u64 w = prev_out->shape.width;
    ts_tensor* jacobian = ts_tensor_create(scratch.arena, (ts_tensor_shape){ w, w, 1 });
    ts_f32* jacobian_data = (ts_f32*)jacobian->data;

    for (ts_u64 x = 0; x < w; x++ ){
        for (ts_u64 y = 0; y < w; y++) {
            if (x == y) {
                jacobian_data[x + y * w] = prev_out_data[x] * (1.0f - prev_out_data[y]);
            } else {
                jacobian_data[x + y * w] = prev_out_data[x] * (-prev_out_data[y]);
            }
        }
    }

    ts_tensor_dot_ip(delta, false, false, delta, jacobian);

    mga_scratch_release(scratch);
}

#endif // TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

