#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// TODO: replace with more official version
// https://en.wikipedia.org/wiki/Box–Muller_transform
f32 _standard_normal() {
    f32 epsilon = 1e-6;
    f32 two_pi = 2.0 * 3.141592653f;

    f32 u1 = epsilon;
    f32 u2 = 0.0f;

    do {
        u1 = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;
    } while (u1 <= epsilon);
    u2 = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;

    f32 mag = sqrtf(-2.0f * logf(u1));
    f32 z0 = mag * cos(two_pi * u2);
    //f32 z1 = mag * sin(two_pi * u2);

    return z0;
}

void _layer_dense_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    u32 in_size = prev_shape.width;
    u32 out_size = desc->dense.size;

    tensor_shape bias_shape = { out_size, 1, 1 };
    tensor_shape weight_shape = { out_size, in_size, 1 };

    out->shape = bias_shape;

    layer_dense_backend* dense = &out->dense_backend;

    dense->bias = tensor_create(arena, bias_shape);
    dense->weight = tensor_create(arena, weight_shape);
    dense->weight_transposed = tensor_create(arena, weight_shape);
    if (out->training_mode) {
        dense->bias_change = (param_change){ 0 };
        dense->weight_change = (param_change){ 0 };
        param_change_create(arena, &dense->bias_change, bias_shape);
        param_change_create(arena, &dense->weight_change, weight_shape);
    }

    // TODO: better weight init

    f32 weight_scale = 1.0f / sqrtf(out_size);
    u64 weight_size = (u64)weight_shape.width * weight_shape.height * weight_shape.depth;
    for (u64 i = 0; i < weight_size; i++) {
        //dense->weight->data[i] = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;
        dense->weight->data[i] = _standard_normal();
        dense->weight->data[i] *= weight_scale;
    }

    tensor_copy_ip(dense->weight_transposed, dense->weight);
    tensor_transpose(dense->weight_transposed);
}
void _layer_dense_feedforward(layer* l, tensor* in_out) {
    layer_dense_backend* dense = &l->dense_backend;

    tensor_dot_ip(in_out, in_out, dense->weight);
    tensor_add_ip(in_out, in_out, dense->bias);
}
void _layer_dense_backprop(layer* l, tensor* delta, tensor* prev_input) {
    layer_dense_backend* dense = &l->dense_backend;

    // Bias change is just delta
    tensor_add_ip(dense->bias_change.change, dense->bias_change.change, delta);

    // Weight change is previous input dotted with delta
    // weight_change += dot(prev_input, delta)
    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor_transpose(prev_input);
    tensor* cur_weight_change = tensor_dot(scratch.arena, prev_input, delta);
    tensor_add_ip(dense->weight_change.change, dense->weight_change.change, cur_weight_change);

    mga_scratch_release(scratch);

    // Delta is updated by weight
    // delta = dot(delta, transpose(weight))
    
    tensor_dot_ip(delta, delta, dense->weight_transposed);
}
void _layer_dense_apply_changes(layer* l, const optimizer* optim) {
    layer_dense_backend* dense = &l->dense_backend;

    param_change_update(optim, dense->weight, &dense->weight_change);
    param_change_update(optim, dense->bias, &dense->bias_change);

    tensor_copy_ip(dense->weight_transposed, dense->weight);
    tensor_transpose(dense->weight_transposed);

    tensor_fill(dense->weight_change.change, 0.0f);
    tensor_fill(dense->bias_change.change, 0.0f);
}
void _layer_dense_delete(layer* l) {
    layer_dense_backend* dense = &l->dense_backend;

    param_change_delete(&dense->weight_change);
    param_change_delete(&dense->bias_change);
}

