#include "layers.h"
#include "layers_internal.h"

#include <stdio.h>
#include <stdlib.h>

void _layer_dropout_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);

    layer_dropout_backend* dropout = &out->dropout_backend;

    dropout->keep_rate = desc->dropout.keep_rate;

    out->shape = prev_shape;
}
void _layer_dropout_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    if (l->training_mode && cache != NULL) {
        f32 keep_rate = l->dropout_backend.keep_rate;

        // Creating dropout tensor
        tensor_shape s = in_out->shape;
        tensor* dropout_tensor = tensor_create(cache->arena, s);

        u64 size = (u64)s.width * s.height * s.depth;
        i32 rand_cutoff = (f32)RAND_MAX * keep_rate;
        for (u64 i = 0; i < size; i++) {
            dropout_tensor->data[i] = rand() > rand_cutoff ? 0.0f : 1.0f;
        }

        // Applying tensor to input
        tensor_component_mul_ip(in_out, in_out, dropout_tensor);
        tensor_scale_ip(in_out, in_out, 1.0f / keep_rate);

        // Saving prev_input and dropout_tensor in cache
        //tensor* input = tensor_copy(cache->arena, in_out, false);
        //layers_cache_push(cache, input);
        layers_cache_push(cache, dropout_tensor);
    }
}
void _layer_dropout_backprop(layer* l, tensor* delta, layers_cache* cache) {
    f32 keep_rate = l->dropout_backend.keep_rate;

    tensor* dropout_tensor = layers_cache_pop(cache);
    //tensor* prev_input = layers_cache_pop(cache);

    /*tensor_shape s = dropout_tensor->shape;
    u64 size = (u64)s.width * s.height * s.depth;

    for (u64 i = 0; i < size; i++) {
        printf("%f ", dropout_tensor->data[i]);
    }*/

    tensor_component_mul_ip(delta, delta, dropout_tensor);
    tensor_scale_ip(delta, delta, 1.0f / keep_rate);

    /*tensor_component_mul_ip(prev_input, prev_input, dropout_tensor);
    tensor_scale_ip(prev_input, prev_input, 1.0f / keep_rate);

    tensor_component_mul_ip(delta, delta, prev_input);*/
}

