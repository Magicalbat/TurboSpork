#include "tensor_internal.h"

#include <float.h>
#include <math.h>
#include <string.h>

#if TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

void _tensor_create_alloc_backend(mg_arena* arena, ts_tensor* out, ts_tensor_shape shape, ts_u64 alloc) {
    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, ts_f32, alloc);
}
void _tensor_destroy_backend(ts_tensor* t) {
    // Nothing to do here because data is all on arenas
    TS_UNUSED(t);
}
void _tensor_copy_backend(ts_tensor* out, const ts_tensor* t, ts_u64 size) {
    memcpy(out->data, t->data, sizeof(ts_f32) * size);
}
void _tensor_fill_backend(ts_tensor* tensor, ts_f32 num) {
    ts_tensor_shape shape = tensor->shape;
    ts_u64 size = (ts_u64)shape.width * shape.height * shape.depth;

    ts_f32* data = (ts_f32*)tensor->data;

    for (ts_u64 i = 0; i < size; i++) {
        data[i] = num;
    }
}
ts_tensor_index _tensor_argmax_backend(const ts_tensor* t) {
    ts_f32* data = (ts_f32*)t->data;

    ts_f32 max_num = -FLT_MAX;
    ts_tensor_index max_index = { 0, 0, 0 };

    for (ts_u64 z = 0; z < t->shape.depth; z++) {
        for (ts_u64 y = 0; y < t->shape.height; y++) {
            for (ts_u64 x = 0; x < t->shape.width; x++) {
                if (data[x + y * t->shape.width + z * t->shape.width * t->shape.height] > max_num) {
                    max_num = data[x + y * t->shape.width + z * t->shape.width * t->shape.height];
                    max_index = (ts_tensor_index){ x, y, z };
                }
            }
        }
    }

    return max_index;
}
ts_b32 _tensor_is_zero(const ts_tensor* t) {
    ts_b32 is_zero = true;

    ts_f32* data = (ts_f32*)t->data;

    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        if (data[i] != 0.0f) {
            is_zero = false;
            break;
        }
    }

    return is_zero;

}
void _tensor_2d_view_backend(ts_tensor* out, const ts_tensor* tensor, ts_u32 z) {
    out->shape = (ts_tensor_shape) {
        .width = tensor->shape.width,
        .height = tensor->shape.height,
        .depth = 1
    };
    out->alloc = (ts_u64)out->shape.width * out->shape.height;

    ts_u64 start_i = (ts_u64)z * tensor->shape.width * tensor->shape.height;

    ts_f32* data = tensor->data;
    out->data = (void*)&data[start_i];
}
// Varients of dot with different transposing
// a_width is after transposing
// lda and ldb are the widths of a and b, before transposing

// Neither are transposed
void _dot_nn(ts_tensor* out, ts_u32 a_width, ts_f32* a_data, ts_u32 lda, ts_f32* b_data, ts_u32 ldb) {
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 y = 0; y < out->shape.height; y++) {
        for (ts_u32 i = 0; i < a_width; i++) {
            // This does not change throughout the inner loop
            ts_f32 a_elem = a_data[(ts_u64)i + (ts_u64)y * lda];
            for (ts_u32 x = 0; x < out->shape.width; x++) {
                out_data[(ts_u64)x + (ts_u64)y * out->shape.width] += a_elem * b_data[(ts_u64)x + (ts_u64)i * ldb];
            }
        }
    }
}

// b is transposed
void _dot_nt(ts_tensor* out, ts_u32 a_width, ts_f32* a_data, ts_u32 lda, ts_f32* b_data, ts_u32 ldb) {
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 y = 0; y < out->shape.height; y++) {
        for (ts_u32 x = 0; x < out->shape.width; x++) {
            ts_f32 sum = 0.0f;
            for (ts_u32 i = 0; i < a_width; i++) {
                sum += a_data[(ts_u64)i + (ts_u64)y * lda] * b_data[(ts_u64)i + (ts_u64)x * ldb];
            }
            out_data[(ts_u64)x + (ts_u64)y * out->shape.width] = sum;
        }
    }
}

// a is transposed
void _dot_tn(ts_tensor* out, ts_u32 a_width, ts_f32* a_data, ts_u32 lda, ts_f32* b_data, ts_u32 ldb) {
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 y = 0; y < out->shape.height; y++) {
        for (ts_u32 i = 0; i < a_width; i++) {
            ts_f32 a_elem = a_data[(ts_u64)y + (ts_u64)i * lda];
            for (ts_u32 x = 0; x < out->shape.width; x++) {
                out_data[(ts_u64)x + (ts_u64)y * out->shape.width] += a_elem * b_data[(ts_u64)x + (ts_u64)i * ldb];
            }
        }
    }
}

// Both are 
void _dot_tt(ts_tensor* out, ts_u32 a_width, ts_f32* a_data, ts_u32 lda, ts_f32* b_data, ts_u32 ldb) {
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 y = 0; y < out->shape.height; y++) {
        for (ts_u32 x = 0; x < out->shape.width; x++) {
            ts_f32 sum = 0.0f;
            for (ts_u32 i = 0; i < a_width; i++) {
                 sum += a_data[(ts_u64)y + (ts_u64)i * lda] * b_data[(ts_u64)i + (ts_u64)x * ldb];
            }
            out_data[(ts_u64)x + (ts_u64)y * out->shape.width] += sum;
        }
    }
}


// Inputs cannot overlap with output
// Output shape should be set up
void _tensor_dot_backend(ts_tensor* out, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b) {
    ts_u32 lda = a->shape.width;
    ts_u32 ldb = b->shape.width;

    ts_f32* a_data = (ts_f32*)a->data;
    ts_f32* b_data = (ts_f32*)b->data;

    _tensor_fill_backend(out, 0.0f);

    ts_u32 a_width = transpose_a ? a->shape.height : a->shape.width;

    if (!transpose_a && !transpose_b) {
        _dot_nn(out, a_width, a_data, lda, b_data, ldb);
    } else if (!transpose_a && transpose_b) {
        _dot_nt(out, a_width, a_data, lda, b_data, ldb);
    } else if (transpose_a && !transpose_b) {
        _dot_tn(out, a_width, a_data, lda, b_data, ldb);
    } else {
        _dot_tt(out, a_width, a_data, lda, b_data, ldb);
    }
}
void _tensor_im2col_backend(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding, ts_u32 x_kernels, ts_u32 y_kernels) {
    ts_tensor_fill(out, 0.0f);

    ts_f32* in_data = (ts_f32*)input->data;
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 z = 0; z < input->shape.depth; z++) {
        for (ts_u32 k = 0; k < kernel_size * kernel_size; k++) {
            ts_u32 x_off = k % kernel_size;
            ts_u32 y_off = k / kernel_size;

            for (ts_u32 y = 0; y < y_kernels; y++) {
                for (ts_u32 x = 0; x < x_kernels; x++) {
                    ts_u32 in_x = x_off + x * stride - padding;
                    ts_u32 in_y = y_off + y * stride - padding;
                    ts_u64 in_index = ((ts_u64)z * input->shape.height + in_y) * input->shape.width + in_x;

                    ts_u32 out_x = y * x_kernels + x;
                    ts_u32 out_y = (z * kernel_size * kernel_size) + k;
                    ts_u64 out_index = (ts_u64)out_y * out->shape.width + out_x;

                    if (in_x < 0 || in_y < 0 || in_x >= input->shape.width || in_y >= input->shape.height) {
                        out_data[out_index] = 0.0f;
                    } else {
                        out_data[out_index] = in_data[in_index];
                    }
                }
            }
        }
    }
}
void _tensor_col2im_backend(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding, ts_u32 x_kernels, ts_u32 y_kernels) {
    ts_tensor_fill(out, 0.0f);

    ts_f32* in_data = (ts_f32*)input->data;
    ts_f32* out_data = (ts_f32*)out->data;

    for (ts_u32 z = 0; z < out->shape.depth; z++) {
        for (ts_u32 k = 0; k < kernel_size * kernel_size; k++) {
            ts_u32 x_off = k % kernel_size;
            ts_u32 y_off = k / kernel_size;

            for (ts_u32 y = 0; y < y_kernels; y++) {
                for (ts_u32 x = 0; x < x_kernels; x++) {
                    ts_u32 in_x = y * x_kernels + x;
                    ts_u32 in_y = (z * kernel_size * kernel_size) + k;
                    ts_u64 in_index = (ts_u64)in_y * input->shape.width + in_x;

                    ts_u32 out_x = x_off + x * stride - padding;
                    ts_u32 out_y = y_off + y * stride - padding;
                    ts_u64 out_index = ((ts_u64)z * out->shape.height + out_y) * out->shape.width + out_x;

                    if (out_x >= 0 && out_x < out->shape.width && out_y >= 0 && out_y < out->shape.height) {
                        out_data[out_index] += in_data[in_index];
                    }
                }
            }
        }
    }
}
// Cannot overlap
void _tensor_transpose_backend(ts_tensor* out, const ts_tensor* t) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* t_data = (ts_f32*)t->data;

    for (ts_u64 x = 0; x < out->shape.width; x++) {
        for (ts_u64 y = 0; y < out->shape.height; y++) {
            out_data[x + y * out->shape.width] = t_data[y + x * t->shape.width];
        }
    }
}
void _tensor_add_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* a_data = (ts_f32*)a->data;
    ts_f32* b_data = (ts_f32*)b->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] + b_data[i];
    }
}
void _tensor_sub_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* a_data = (ts_f32*)a->data;
    ts_f32* b_data = (ts_f32*)b->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] - b_data[i];
    }
}
void _tensor_component_mul_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* a_data = (ts_f32*)a->data;
    ts_f32* b_data = (ts_f32*)b->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] * b_data[i];
    }
}
void _tensor_component_div_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* a_data = (ts_f32*)a->data;
    ts_f32* b_data = (ts_f32*)b->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = a_data[i] / b_data[i];
    }
}
void _tensor_add_all_backend(ts_tensor* out, const ts_tensor* t, ts_f32 x) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* in_data = (ts_f32*)t->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = in_data[i] + x;
    }
}
void _tensor_scale_backend(ts_tensor* out, const ts_tensor* t, ts_f32 s) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* t_data = (ts_f32*)t->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = t_data[i] * s;
    }
}
void _tensor_sqrt_backend(ts_tensor* out, const ts_tensor* t) {
    ts_f32* out_data = (ts_f32*)out->data;
    ts_f32* t_data = (ts_f32*)t->data;

    ts_u64 size = (ts_u64)out->shape.width * out->shape.height * out->shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        out_data[i] = sqrtf(t_data[i]);
    }
}
void _tensor_get_data_backend(ts_f32* out, const ts_tensor* t) {
    ts_f32* t_data = (ts_f32*)t->data;

    if (out == t_data)
        return;

    memcpy(out, t_data, sizeof(ts_f32) * t->shape.width * t->shape.height * t->shape.depth);
}
void _tensor_set_data_backend(ts_tensor* t, ts_f32* data) {
    ts_f32* t_data = (ts_f32*)t->data;

    if (data == t_data)
        return;

    memcpy(t_data, data, sizeof(ts_f32) * t->shape.width * t->shape.height * t->shape.depth);
}

#endif // TS_TENSOR_BACKEND == TS_TENSOR_BACKEND_CPU

