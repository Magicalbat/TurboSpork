#include "layers.h"
#include "layers_internal.h"

#include <math.h>

void _layer_norm_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    out->shape = prev_shape;

    out->norm_backend.epsilon = desc->norm.epsilon;
}
void _layer_norm_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    ts_u64 layer_size = (ts_u64)in_out->shape.width * in_out->shape.height * in_out->shape.depth;

    float mean = 0.0f;
    for (ts_u64 i = 0; i < layer_size; i++) {
        mean += in_out->data[i];
    }
    mean /= layer_size;

    float std_dev = 0.0f;
    for (ts_u64 i = 0; i < layer_size; i++) {
        std_dev += (in_out->data[i] - mean) * (in_out->data[i] - mean);
    }
    std_dev = sqrtf((std_dev / layer_size) + l->norm_backend.epsilon);

    if (cache != NULL && l->training_mode) {
        ts_tensor* stdv = ts_tensor_create(cache->arena, (ts_tensor_shape){ 1, 1, 1 });
        stdv->data[0] = std_dev;
        
        ts_layers_cache_push(cache, stdv);
    }

    for (ts_u64 i = 0; i < layer_size; i++) {
        in_out->data[i] = (in_out->data[i] - mean) / std_dev;
    }
}
void _layer_norm_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    TS_UNUSED(l);

    ts_tensor* stdv = ts_layers_cache_pop(cache);

    ts_tensor_scale_ip(delta, delta, 1.0f / stdv->data[0]);
}

