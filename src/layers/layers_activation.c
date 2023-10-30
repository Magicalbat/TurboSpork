#include "layers_internal.h"

#include <stdio.h>
#include <math.h>

typedef void (_activ_func)(tensor*);
typedef void (_activ_grad)(tensor*, tensor*);

typedef struct {
    _activ_func* func;
    _activ_grad* grad;
} _activation;

static void _null_func(tensor* t);
static void _null_grad(tensor* prev_in, tensor* delta);
static void _sigmoid_func(tensor* t);
static void _sigmoid_grad(tensor* prev_in, tensor* delta);
static void _tanh_func(tensor* t);
static void _tanh_grad(tensor* prev_in, tensor* delta);
static void _relu_func(tensor* t);
static void _relu_grad(tensor* prev_in, tensor* delta);
static void _leaky_relu_func(tensor* t);
static void _leaky_relu_grad(tensor* prev_in, tensor* delta);
static void _softmax_func(tensor* t);
static void _softmax_grad(tensor* prev_in, tensor* delta);

static _activation _activations[ACTIVATION_COUNT] = {
    [ACTIVATION_NULL] = { _null_func, _null_grad },
    [ACTIVATION_SIGMOID] = { _sigmoid_func, _sigmoid_grad },
    [ACTIVATION_TANH] = { _tanh_func, _tanh_grad },
    [ACTIVATION_RELU] = { _relu_func, _relu_grad },
    [ACTIVATION_LEAKY_RELU] = { _leaky_relu_func, _leaky_relu_grad },
    [ACTIVATION_SOFTMAX] = { _softmax_func, _softmax_grad },
};

void _layer_activation_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);

    layer_activation_backend* activ = &out->activation_backend;

    activ->type = desc->activation.type;

    if (activ->type >= ACTIVATION_COUNT) {
        fprintf(stderr, "Invalid activation type\n");

        activ->type = ACTIVATION_NULL;
    }

    out->shape = prev_shape;
}
void _layer_activation_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_activation_backend* activ = &l->activation_backend;

    if (cache != NULL && l->training_mode) {
        tensor* input = tensor_copy(cache->arena, in_out, false);
        layers_cache_push(cache, input);
    }

    _activations[activ->type].func(in_out);
}
void _layer_activation_backprop(layer* l, tensor* delta, layers_cache* cache) {
    layer_activation_backend* activ = &l->activation_backend;

    tensor* prev_input = layers_cache_pop(cache);

    _activations[activ->type].grad(prev_input, delta);
}

static void _null_func(tensor* t) {
    UNUSED(t);
}
static void _null_grad(tensor* prev_in, tensor* delta) {
    UNUSED(prev_in);
    UNUSED(delta);
}

#define _LOOP_T(t) u64 _size = (u64)t->shape.width * t->shape.height * t->shape.depth; \
    for (u64 i = 0; i < _size; i++) 

static void _sigmoid_func(tensor* t) {
    _LOOP_T(t) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}
static void _sigmoid_grad(tensor* prev_in, tensor* delta) {
    _LOOP_T(prev_in) {
        f32 s = 1.0f / (1.0f + expf(-prev_in->data[i]));
        prev_in->data[i] = s * (1.0f - s);
    }

    tensor_component_mul_ip(delta, delta, prev_in);
}

static void _tanh_func(tensor* t) {
    _LOOP_T(t) {
        t->data[i] = tanh(t->data[i]);
    }
}
static void _tanh_grad(tensor* prev_in, tensor* delta) {
    _LOOP_T(prev_in) {
        f32 tnh = tanh(prev_in->data[i]);
        prev_in->data[i] = 1.0f - tnh * tnh;
    }

    tensor_component_mul_ip(delta, delta, prev_in);
}

static void _relu_func(tensor* t) {
    _LOOP_T(t) {
        t->data[i] = t->data[i] > 0.0f ? t->data[i] : 0.0f;
    }
}
static void _relu_grad(tensor* prev_in, tensor* delta) {
    _LOOP_T(prev_in) {
        prev_in->data[i] = (prev_in->data[i] > 0.0f);
    }

    tensor_component_mul_ip(delta, delta, prev_in);
}

static void _leaky_relu_func(tensor* t) {
    _LOOP_T(t) {
        t->data[i] = t->data[i] > 0.0f ? t->data[i] : t->data[i] * 0.01f;
    }
}
static void _leaky_relu_grad(tensor* prev_in, tensor* delta) {
    _LOOP_T(prev_in) {
        prev_in->data[i] = prev_in->data[i] > 0.0f ? 1.0f : 0.01f;
    }

    tensor_component_mul_ip(delta, delta, prev_in);
}

// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative 
static void _softmax_func(tensor* t) {
    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;

    // Computing max for stability
    f32 max_num = t->data[0];
    for (u64 i = 0; i < size; i++) {
        if (t->data[i] > max_num) {
            max_num = t->data[i];
        }
    }

    // Exponentiation and exponential sum
    f32 exp_sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        t->data[i] = expf(t->data[i] - max_num);
        exp_sum += t->data[i];
    }

    for (u64 i = 0; i < size; i++) {
        t->data[i] /= exp_sum;
    }
}
static void _softmax_grad(tensor* prev_in, tensor* delta) {
    _softmax_func(prev_in);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    u64 w = prev_in->shape.width;
    tensor* jacobian = tensor_create(scratch.arena, (tensor_shape){ w, w, 1 });
    for (u64 x = 0; x < w; x++ ){
        for (u64 y = 0; y < w; y++) {
            if (x == y) {
                jacobian->data[x + y * w] = prev_in->data[x] * (1.0f - prev_in->data[y]);
            } else {
                jacobian->data[x + y * w] = prev_in->data[x] * (-prev_in->data[y]);
            }
        }
    }

    tensor_dot_ip(delta, delta, jacobian);

    mga_scratch_release(scratch);
}


