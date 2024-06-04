#ifndef TENSOR_INTERNAL_H
#define TENSOR_INTERNAL_H

#include "tensor.h"

// All of these are implemented by different backends
// All of these assume arguments are valid

void _tensor_create_alloc_backend(mg_arena* arena, ts_tensor* out, ts_tensor_shape shape, ts_u64 alloc);
void _tensor_destroy_backend(ts_tensor* t);
void _tensor_copy_backend(ts_tensor* out, const ts_tensor* t, ts_u64 size);
void _tensor_fill_backend(ts_tensor* tensor, ts_f32 num);
ts_tensor_index _tensor_argmax_backend(const ts_tensor* t);
ts_b32 _tensor_is_zero(const ts_tensor* t);
void _tensor_2d_view_backend(ts_tensor* out, const ts_tensor* tensor, ts_u32 z);
// Inputs cannot overlap with output
void _tensor_dot_backend(ts_tensor* out, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b);
void _tensor_im2col_backend(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding, ts_u32 x_kernels, ts_u32 y_kernels);
void _tensor_col2im_backend(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding, ts_u32 x_kernels, ts_u32 y_kernels);
// Cannot overlap
void _tensor_transpose_backend(ts_tensor* out, const ts_tensor* t);
void _tensor_add_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
void _tensor_sub_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
void _tensor_component_mul_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
void _tensor_component_div_backend(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
void _tensor_sqrt_backend(ts_tensor* out, const ts_tensor* t);
void _tensor_scale_backend(ts_tensor* out, const ts_tensor* t, ts_f32 s);
void _tensor_get_data_backend(ts_f32* out, const ts_tensor* t);
void _tensor_set_data_backend(ts_tensor* t, ts_f32* data);

#endif // TENSOR_INTERNAL_H

