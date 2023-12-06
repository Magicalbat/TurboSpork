#include "layers.h"
#include "layers_internal.h"
#include "prng.h"

#include <stdio.h>
#include <stdlib.h>

void _layer_dropout_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    _layer_dropout_backend* dropout = &out->dropout_backend;

    dropout->keep_rate = desc->dropout.keep_rate;

    out->shape = prev_shape;
}
void _layer_dropout_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    if (l->training_mode && cache != NULL) {
        ts_f32 keep_rate = l->dropout_backend.keep_rate;

        // Creating dropout ts_tensor
        ts_tensor_shape s = in_out->shape;
        ts_tensor* dropout_tensor = ts_tensor_create(cache->arena, s);

        ts_u64 size = (ts_u64)s.width * s.height * s.depth;
        for (ts_u64 i = 0; i < size; i++) {
            dropout_tensor->data[i] = ts_prng_rand_f32() > keep_rate ? 0.0f : 1.0f;
        }

        // Applying tensor to input
        ts_tensor_component_mul_ip(in_out, in_out, dropout_tensor);
        ts_tensor_scale_ip(in_out, in_out, 1.0f / keep_rate);

        // Saving dropout_tensor in cache
        ts_layers_cache_push(cache, dropout_tensor);
    }
}
void _layer_dropout_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    ts_f32 keep_rate = l->dropout_backend.keep_rate;

    ts_tensor* dropout_tensor = ts_layers_cache_pop(cache);

    ts_tensor_component_mul_ip(delta, delta, dropout_tensor);
    ts_tensor_scale_ip(delta, delta, 1.0f / keep_rate);
}

