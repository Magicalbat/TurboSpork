#include "layers.h"
#include "layers_internal.h"

#include <stdlib.h>
#include <math.h>

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

    f32 weight_scale = 1.0f / sqrtf(out_size);
    u64 weight_size = (u64)weight_shape.width * weight_shape.height * weight_shape.depth;
    for (u64 i = 0; i < weight_size; i++) {
        dense->weight->data[i] = prng_std_norm();
        dense->weight->data[i] *= weight_scale;
    }

    tensor_copy_ip(dense->weight_transposed, dense->weight);
    tensor_transpose(dense->weight_transposed);
}
#include <stdio.h>
void _layer_dense_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    layer_dense_backend* dense = &l->dense_backend;

    if (cache != NULL && l->training_mode) {
        tensor* input = tensor_copy(cache->arena, in_out, false);
        layers_cache_push(cache, input);
    }

    tensor_dot_ip(in_out, in_out, dense->weight);
    tensor_add_ip(in_out, in_out, dense->bias);
}
void _layer_dense_backprop(layer* l, tensor* delta, layers_cache* cache) {
    layer_dense_backend* dense = &l->dense_backend;

    // Bias change is just delta
    tensor_add_ip(dense->bias_change.change, dense->bias_change.change, delta);

    // Weight change is previous input dotted with delta
    // weight_change += dot(prev_input, delta)
    mga_temp scratch = mga_scratch_get(&cache->arena, 1);

    tensor* prev_input = layers_cache_pop(cache);

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
void _layer_dense_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {
    layer_dense_backend* dense = &l->dense_backend;

    string8 weight_name = str8_pushf(arena, "dense_weight_%u", index);
    string8 bias_name = str8_pushf(arena, "dense_bias_%u", index);

    tensor_list_push(arena, list, dense->weight, weight_name);
    tensor_list_push(arena, list, dense->bias, bias_name);
}
void _layer_dense_load(layer* l, const tensor_list* list, u32 index) {
    layer_dense_backend* dense = &l->dense_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 weight_name = str8_pushf(scratch.arena, "dense_weight_%u", index);
    string8 bias_name = str8_pushf(scratch.arena, "dense_bias_%u", index);

    tensor* loaded_weight = tensor_list_get(list, weight_name);
    tensor* loaded_bias = tensor_list_get(list, bias_name);

    tensor_copy_ip(dense->weight, loaded_weight);
    tensor_copy_ip(dense->bias, loaded_bias);

    tensor_copy_ip(dense->weight_transposed, loaded_weight);
    tensor_transpose(dense->weight_transposed);

    mga_scratch_release(scratch);
}

