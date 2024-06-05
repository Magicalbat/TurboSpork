#include "layers.h"
#include "layers_internal.h"

#include <math.h>

// Returns standard deviation
ts_f32 _norm_backend(ts_tensor* t, ts_f32 epsilon);

void _layer_norm_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    out->shape = prev_shape;

    out->norm_backend.epsilon = desc->norm.epsilon;
}
void _layer_norm_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    ts_f32 std_dev = _norm_backend(in_out, l->norm_backend.epsilon);

    // TODO: store this without the tensor cache
    // Very dumb with GPU backend
    if (cache != NULL && l->training_mode) {
        ts_tensor* stdv = ts_tensor_create(cache->arena, (ts_tensor_shape){ 1, 1, 1 });
        ts_tensor_set_data(stdv, &std_dev);
        
        ts_layers_cache_push(cache, stdv);
    }
}
void _layer_norm_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    TS_UNUSED(l);

    ts_tensor* stdv = ts_layers_cache_pop(cache);
    ts_f32 std_dev = 1.0f;
    ts_tensor_get_data(&std_dev, stdv);

    ts_tensor_scale_ip(delta, delta, 1.0f / std_dev);
}

#if TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

ts_f32 _norm_backend(ts_tensor* t, ts_f32 epsilon) {
    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;

    ts_f32* data = (ts_f32*)t->data;

    float mean = 0.0f;
    for (ts_u64 i = 0; i < size; i++) {
        mean += data[i];
    }
    mean /= size;

    float std_dev = 0.0f;
    for (ts_u64 i = 0; i < size; i++) {
        std_dev += (data[i] - mean) * (data[i] - mean);
    }
    std_dev = sqrtf((std_dev / size) + epsilon);

    for (ts_u64 i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std_dev;
    }

    return std_dev;
}

#endif // TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

