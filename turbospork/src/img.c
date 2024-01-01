#include "img.h"
#include "err.h"
#include "prng.h"

#include <math.h>
#include <string.h>

typedef struct {
    ts_f32 x, y;
} _vec2;

// Returns zero if out of bounds
#define _GET_PIXEL(x, y, z) \
    ((x) < 0 || (x) >= width || (y) < 0 || (y) >= height) ? \
    0 : img_data[((ts_u64)(z) * height + (y)) * width + (x)]

ts_f32 _sample_img(ts_f32* img_data, ts_u32 width, ts_u32 height, _vec2 pos, ts_u32 z, ts_img_sample_type sample_type) {
    switch (sample_type) {
        case TS_SAMPLE_NEAREST: {
            ts_i64 x = floorf(pos.x);
            ts_i64 y = floorf(pos.y);

            return _GET_PIXEL(x, y, z);
        } break;
        case TS_SAMPLE_BILINEAR: {
            //pos.x -= 0.5f;
            //pos.y -= 0.5f;

            ts_i64 x = floorf(pos.x);
            ts_i64 y = floorf(pos.y);

            ts_f32 p0 = _GET_PIXEL(x    , y    , z);
            ts_f32 p1 = _GET_PIXEL(x + 1, y    , z);
            ts_f32 p2 = _GET_PIXEL(x    , y + 1, z);
            ts_f32 p3 = _GET_PIXEL(x + 1, y + 1, z);

            // Lerping, where t is the decimal part of the x pos
            ts_f32 top_p = p0 + (p1 - p0) * (pos.x - x);
            ts_f32 bot_p = p2 + (p3 - p2) * (pos.x - x);

            // Lerping, where t is the decimal part of the y pos
            ts_f32 p = top_p + (bot_p - top_p) * (pos.y - y);

            return p;
        } break;
        default: break;
    }

    return 0.0f;
}

ts_b32 ts_img_transform_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, const ts_img_mat3* mat) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot transform image: out and/or input is NULL");
        return false;
    }
    if (mat == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot transform image: mat is NULL");
        return false;
    }

    ts_u64 data_size = (ts_u64)input->shape.width * input->shape.height * input->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot transform image: not enough space in out");
        #endif

        return false;
    }

    // Calculating the inverse matrix
    // The pixel points need to be transformed by the inverse
    // so that the image is transformed by the matrix

    // If you know a better way to do this,
    // please make an issue or submit a pull request

    ts_f32 inverse_mat[9] = { 0 };

    // Renaming so it is easier to type
    const ts_f32* m = mat->m;
    ts_f32 mat_det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);

    if (fabsf(mat_det) < 1e-6f) {
        TS_ERR(TS_ERR_MATH, "Cannot transform image: determinant of transformation matrix is near zero");
        return false;
    }

    ts_f32 inv_det = 1.0f / mat_det;

    inverse_mat[0] = (m[4] * m[8] - m[5] * m[7]) * inv_det;
    inverse_mat[1] = (m[2] * m[7] - m[1] * m[8]) * inv_det;
    inverse_mat[2] = (m[1] * m[5] - m[2] * m[4]) * inv_det;
    inverse_mat[3] = (m[5] * m[6] - m[3] * m[8]) * inv_det;
    inverse_mat[4] = (m[0] * m[8] - m[2] * m[6]) * inv_det;
    inverse_mat[5] = (m[2] * m[3] - m[0] * m[5]) * inv_det;
    inverse_mat[6] = (m[3] * m[7] - m[4] * m[6]) * inv_det;
    inverse_mat[7] = (m[1] * m[6] - m[0] * m[7]) * inv_det;
    inverse_mat[8] = (m[0] * m[4] - m[1] * m[3]) * inv_det;

    out->shape = input->shape;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Checking if out and input overlap
    // Copying the data if they do
    ts_f32* img_data = input->data;
    if (out->data == input->data) {
        ts_u64 data_size = (ts_u64)input->shape.width * input->shape.height * input->shape.depth;
        img_data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, data_size);
        memcpy(img_data, input->data, data_size * sizeof(ts_f32));
    }

    _vec2 offset = {
        (ts_f32)input->shape.width / 2,
        (ts_f32)input->shape.height / 2
    };

    for (ts_u32 z = 0; z < input->shape.depth; z++) {
        for (ts_u32 y = 0; y < input->shape.height; y++) {
            for (ts_u32 x = 0; x < input->shape.width; x++) {
                // Offset to center
                _vec2 pos = {
                    (ts_f32)x - offset.x,
                    (ts_f32)y - offset.y,
                };

                _vec2 out_pos = {
                    (pos.x * inverse_mat[0]) + (pos.y * inverse_mat[1]) + inverse_mat[2],
                    (pos.x * inverse_mat[3]) + (pos.y * inverse_mat[4]) + inverse_mat[5],
                    // Other values in matrix are ignored
                };

                // Undoing centering
                out_pos.x += offset.x;
                out_pos.y += offset.y;

                ts_u64 index = ((ts_u64)z * out->shape.height + y) * out->shape.width + x;
                out->data[index] = _sample_img(img_data, input->shape.width, input->shape.height, out_pos, z, sample_type);
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}

ts_b32 ts_img_translate_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_off, ts_f32 y_off) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot translate image with NULL input and/or output");
        return false;
    }

    ts_img_mat3 mat = {
        {
            1, 0, x_off,
            0, 1, y_off,
            0, 0, 1
        }
    };

    return ts_img_transform_ip(out, input, sample_type, &mat);
}

ts_b32 ts_img_scale_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_scale, ts_f32 y_scale) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot scale image with NULL input and/or output");
        return false;
    }

    ts_img_mat3 mat = {
        {
            x_scale, 0, 0,
            0, y_scale, 0,
            0, 0, 1
        }
    };

    return ts_img_transform_ip(out, input, sample_type, &mat);
}

ts_b32 ts_img_rotate_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 theta) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot rotate image with NULL input and/or output");
        return false;
    }

    ts_f32 sin_theta = sinf(theta);
    ts_f32 cos_theta = cosf(theta);

    ts_img_mat3 mat = {
        {
            cos_theta, -sin_theta, 0,
            sin_theta,  cos_theta, 0,
            0, 0, 1
        }
    };

    return ts_img_transform_ip(out, input, sample_type, &mat);
}

ts_b32 ts_img_shear_ip(ts_tensor* out, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_shear, ts_f32 y_shear) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot shear image with NULL input and/or output");
        return false;
    }

    ts_img_mat3 mat = {
        {
            1, x_shear, 0,
            y_shear, 1, 0,
            0, 0, 1
        }
    };

    return ts_img_transform_ip(out, input, sample_type, &mat);
}

ts_b32 ts_img_add_noise_ip(ts_tensor* out, const ts_tensor* input, ts_f32 noise_rate) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add noise to image with NULL input and/or output");
        return false;
    }

    ts_u64 data_size = (ts_u64)input->shape.width * input->shape.height * input->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot add noise to image: not enough space in out");
        #endif

        return false;
    }

    ts_tensor_copy_ip(out, input);

    for (ts_u32 y = 0; y < out->shape.height; y++) {
        for (ts_u32 x = 0; x < out->shape.width; x++) {
            if (ts_prng_rand_f32() < noise_rate) {
                // I know this is bad for cache,
                // but this is not as performance
                // critical as something like the dot product

                for (ts_u32 z = 0; z < out->shape.depth; z++) {
                    ts_u64 index = ((ts_u64)z * out->shape.height + y) * out->shape.width + x;

                    out->data[index] = ts_prng_rand_f32();
                }
            }
        }
    }

    return true;
}

ts_tensor* ts_img_transform(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, const ts_img_mat3* mat) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot transform NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_transform_ip(out, input, sample_type, mat);

    return out;
}

ts_tensor* ts_img_translate(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_off, ts_f32 y_off) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot translate NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_translate_ip(out, input, sample_type, x_off, y_off);

    return out;
}

ts_tensor* ts_img_scale(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_scale, ts_f32 y_scale) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot scale NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_scale_ip(out, input, sample_type, x_scale, y_scale);

    return out;
}

ts_tensor* ts_img_rotate(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 theta) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot rotate NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_rotate_ip(out, input, sample_type, theta);

    return out;
}

ts_tensor* ts_img_shear(mg_arena* arena, const ts_tensor* input, ts_img_sample_type sample_type, ts_f32 x_shear, ts_f32 y_shear) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot shear NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_shear_ip(out, input, sample_type, x_shear, y_shear);

    return out;
}

ts_tensor* ts_img_add_noise(mg_arena* arena, const ts_tensor* input, ts_f32 noise_rate) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add noise to NULL image");
        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, input->shape);

    ts_img_add_noise_ip(out, input, noise_rate);

    return out;
}

