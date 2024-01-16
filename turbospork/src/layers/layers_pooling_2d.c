#include "layers.h"
#include "layers_internal.h"

#include <float.h>
#include <stdio.h>

typedef void (_pooling_func)(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta);

void _pool_null(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta);
void _pool_max(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta);
void _pool_avg(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta);

_pooling_func* _pooling_funcs[TS_POOLING_COUNT] = {
    [TS_POOLING_NULL] = _pool_null,
    [TS_POOLING_MAX] = _pool_max,
    [TS_POOLING_AVG] = _pool_avg,
};

void _layer_pooling_2d_create(mg_arena* arena, ts_layer* out, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    TS_UNUSED(arena);

    ts_layer_pooling_2d_backend* pooling = &out->pooling_2d_backend;

    if (desc->pooling_2d.type >= TS_POOLING_COUNT) {
        fprintf(stderr, "Invalid type for pooling layer\n");

        return;
    }

    pooling->input_shape = prev_shape;
    pooling->pool_size = desc->pooling_2d.pool_size;
    pooling->type = desc->pooling_2d.type;

    ts_tensor_shape out_shape = {
        prev_shape.width / pooling->pool_size.width,
        prev_shape.height / pooling->pool_size.height,
        prev_shape.depth,
    };

    out->shape = out_shape;
}

// TODO: Type check here and in activation?
void _layer_pooling_2d_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    ts_layer_pooling_2d_backend* pooling = &l->pooling_2d_backend;

    ts_tensor* delta = NULL;

    if (cache != NULL) {
        delta = ts_tensor_create(cache->arena, in_out->shape);
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_tensor* in_copy = ts_tensor_copy(scratch.arena, in_out, false);
    in_out->shape = l->shape;

    ts_tensor in_copy_view = { 0 };
    ts_tensor out_view = { 0 };
    ts_tensor delta_view = { 0 };

    for (ts_u32 z = 0; z < l->shape.depth; z++) {
        ts_tensor_2d_view(&in_copy_view, in_copy, z);
        ts_tensor_2d_view(&out_view, in_out, z);

        ts_tensor* delta_ptr = NULL;

        if (cache != NULL) {
            ts_tensor_2d_view(&delta_view, delta, z);
            delta_ptr = &delta_view;
        }

        _pooling_funcs[pooling->type](&in_copy_view, &out_view, pooling->pool_size, delta_ptr);
    }

    mga_scratch_release(scratch);

    if (cache != NULL) {
        ts_layers_cache_push(cache, delta);
    }
}

// Prevents indexing input out of bounds
#define _CLIP_POOL_SIZE(p_w, p_h, i_x, i_y, in_shape) do {\
        if (i_x + p_w > in_shape.width) { p_w = i_x + p_w - in_shape.width; } \
        if (i_y + p_h > in_shape.height) { p_h = i_y + p_h - in_shape.height; } \
    } while (0)


void _layer_pooling_2d_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    ts_layer_pooling_2d_backend* pooling = &l->pooling_2d_backend;

    // Expanding delta to input shape
    {
        mga_temp scratch = mga_scratch_get(NULL, 0);

        ts_tensor* delta_copy = ts_tensor_copy(scratch.arena, delta, false);

        delta->shape = pooling->input_shape;

        for (ts_u64 z = 0; z < delta->shape.depth; z++) {
            for (ts_u64 i_y = 0; i_y < delta_copy->shape.height; i_y++) {
                for (ts_u64 i_x = 0; i_x < delta_copy->shape.width; i_x++) {
                    // Starting delta coords
                    ts_u64 d_x = i_x * pooling->pool_size.width;
                    ts_u64 d_y = i_y * pooling->pool_size.height;

                    ts_u64 orig_delta_index = i_x + i_y * delta_copy->shape.width +
                        z * delta_copy->shape.width * delta_copy->shape.height;
                    ts_f32 orig_delta_value = delta_copy->data[orig_delta_index];

                    ts_u32 p_w = pooling->pool_size.width;
                    ts_u32 p_h = pooling->pool_size.height;

                    _CLIP_POOL_SIZE(p_w, p_h, i_x, i_y, delta->shape);

                    ts_u64 z_off = z * delta->shape.width * delta->shape.height;
                    for (ts_u32 x = 0; x < p_w; x++) {
                        for (ts_u32 y = 0; y < p_h; y++) {
                            ts_u64 index = (d_x + x) + (d_y + y) * delta->shape.width + z_off;

                            delta->data[index] = orig_delta_value;
                        }
                    }
                }
            }
        }

        mga_scratch_release(scratch);
    }

    // Multiplying delta by pool_delta (computed in feedforward)
    ts_tensor* pool_delta = ts_layers_cache_pop(cache);
    ts_tensor_component_mul_ip(delta, delta, pool_delta);
}

void _pool_null(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta) {
    TS_UNUSED(in);
    TS_UNUSED(out);
    TS_UNUSED(pool_size);
    TS_UNUSED(delta);
}

void _pool_max(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta) {
    // Looping through output coords
    for (ts_u64 o_y = 0; o_y < out->shape.height; o_y++) {
        for (ts_u64 o_x = 0; o_x < out->shape.width; o_x++) {
            // Starting input coords
            ts_u64 i_x = o_x * pool_size.width;
            ts_u64 i_y = o_y * pool_size.height;

            ts_u32 p_w = pool_size.width;
            ts_u32 p_h = pool_size.height;

            _CLIP_POOL_SIZE(p_w, p_h, i_x, i_y, in->shape);

            // Max num in pool
            ts_f32 max_num = -FLT_MAX;
            ts_u64 max_index = 0;
            
            for (ts_u32 x = 0; x < p_w; x++) {
                for (ts_u32 y = 0; y < p_h; y++) {
                    ts_u64 index = (i_x + x) + (i_y + y) * in->shape.width;

                    if (in->data[index] > max_num) {
                        max_num = in->data[index];
                        max_index = index;
                    }
                }
            }

            out->data[o_x + o_y * out->shape.width] = max_num;

            if (delta != NULL) {
                delta->data[max_index] = 1.0f;
            }
        }
    }
}
void _pool_avg(const ts_tensor* in, ts_tensor* out, ts_tensor_shape pool_size, ts_tensor* delta) {
    // Looping through output coords
    for (ts_u64 o_y = 0; o_y < out->shape.height; o_y++) {
        for (ts_u64 o_x = 0; o_x < out->shape.width; o_x++) {
            // Starting input coords
            ts_u64 i_x = o_x * pool_size.width;
            ts_u64 i_y = o_y * pool_size.height;

            ts_u32 p_w = pool_size.width;
            ts_u32 p_h = pool_size.height;

            _CLIP_POOL_SIZE(p_w, p_h, i_x, i_y, in->shape);

            ts_f32 sum = 0.0f;

            for (ts_u32 x = 0; x < p_w; x++) {
                for (ts_u32 y = 0; y < p_h; y++) {
                    ts_u64 index = (i_x + x) + (i_y + y) * in->shape.width;

                    sum += in->data[index];
                }
            }

            out->data[o_x + o_y * out->shape.width] = sum / (pool_size.width * pool_size.height);

            if (delta == NULL) {
                continue;
            }

            for (ts_u32 x = 0; x < p_w; x++) {
                for (ts_u32 y = 0; y < p_h; y++) {
                    ts_u64 index = (i_x + x) + (i_y + y) * in->shape.width;

                    delta->data[index] = 1.0f / (pool_size.width * pool_size.height);
                }
            }
        }
    }
}

