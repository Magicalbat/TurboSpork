#include "layers.h"
#include "layers_internal.h"

#include <stdlib.h>
#include <math.h>

void _layer_dense_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    ts_u32 in_size = prev_shape.width;
    ts_u32 out_size = desc->dense.size;

    ts_tensor_shape bias_shape = { out_size, 1, 1 };
    ts_tensor_shape weight_shape = { out_size, in_size, 1 };

    out->shape = bias_shape;

    _layer_dense_backend* dense = &out->dense_backend;

    dense->bias = ts_tensor_create(arena, bias_shape);
    dense->weight = ts_tensor_create(arena, weight_shape);
    dense->weight_transposed = ts_tensor_create(arena, weight_shape);

    if (out->training_mode) {
        ts_param_change_create(arena, &dense->bias_change, bias_shape);
        ts_param_change_create(arena, &dense->weight_change, weight_shape);
    }

    ts_param_init(dense->bias, desc->dense.bias_init, in_size, out_size);
    ts_param_init(dense->weight, desc->dense.weight_init, in_size, out_size);

    ts_tensor_copy_ip(dense->weight_transposed, dense->weight);
    ts_tensor_transpose(dense->weight_transposed);
}
void _layer_dense_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    _layer_dense_backend* dense = &l->dense_backend;

    if (cache != NULL && l->training_mode) {
        ts_tensor* input = ts_tensor_copy(cache->arena, in_out, false);
        ts_layers_cache_push(cache, input);
    }

    ts_tensor_dot_ip(in_out, in_out, dense->weight);
    ts_tensor_add_ip(in_out, in_out, dense->bias);
}
void _layer_dense_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    _layer_dense_backend* dense = &l->dense_backend;

    // Bias change is just delta
    ts_param_change_add(&dense->bias_change, delta);

    // Weight change is previous input dotted with delta
    // weight_change += dot(prev_input, delta)
    mga_temp scratch = mga_scratch_get(&cache->arena, 1);

    ts_tensor* prev_input = ts_layers_cache_pop(cache);

    ts_tensor_transpose(prev_input);
    ts_tensor* cur_weight_change = ts_tensor_dot(scratch.arena, prev_input, delta);
    ts_param_change_add(&dense->weight_change, cur_weight_change);

    mga_scratch_release(scratch);

    // Delta is updated by weight
    // delta = dot(delta, transpose(weight))
    ts_tensor_dot_ip(delta, delta, dense->weight_transposed);
}
void _layer_dense_apply_changes(ts_layer* l, const ts_optimizer* optim) {
    _layer_dense_backend* dense = &l->dense_backend;

    ts_param_change_update(optim, dense->weight, &dense->weight_change);
    ts_param_change_update(optim, dense->bias, &dense->bias_change);

    ts_tensor_copy_ip(dense->weight_transposed, dense->weight);
    ts_tensor_transpose(dense->weight_transposed);
}
void _layer_dense_delete(ts_layer* l) {
    _layer_dense_backend* dense = &l->dense_backend;

    if (l->training_mode) {
        ts_param_change_delete(&dense->weight_change);
        ts_param_change_delete(&dense->bias_change);
    }
}
// TODO: Make noe function to get weight and bias names
void _layer_dense_save(mg_arena* arena, ts_tensor_list* list, ts_layer* l, ts_u32 index) {
    _layer_dense_backend* dense = &l->dense_backend;

    ts_string8 weight_name = ts_str8_pushf(arena, "dense_weight_%u", index);
    ts_string8 bias_name = ts_str8_pushf(arena, "dense_bias_%u", index);

    ts_tensor_list_push(arena, list, dense->weight, weight_name);
    ts_tensor_list_push(arena, list, dense->bias, bias_name);
}
void _layer_dense_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index) {
    _layer_dense_backend* dense = &l->dense_backend;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 weight_name = ts_str8_pushf(scratch.arena, "dense_weight_%u", index);
    ts_string8 bias_name = ts_str8_pushf(scratch.arena, "dense_bias_%u", index);

    ts_tensor* loaded_weight = ts_tensor_list_get(list, weight_name);
    ts_tensor* loaded_bias = ts_tensor_list_get(list, bias_name);

    ts_tensor_copy_ip(dense->weight, loaded_weight);
    ts_tensor_copy_ip(dense->bias, loaded_bias);

    ts_tensor_copy_ip(dense->weight_transposed, loaded_weight);
    ts_tensor_transpose(dense->weight_transposed);

    mga_scratch_release(scratch);
}

