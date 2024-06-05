#include "tensor.h"

#include "tensor_internal.h"

#include "os.h"
#include "err.h"

#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

ts_b32 ts_tensor_index_eq(ts_tensor_index a, ts_tensor_index b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
ts_b32 ts_tensor_shape_eq(ts_tensor_shape a, ts_tensor_shape b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

ts_tensor* ts_tensor_create(mg_arena* arena, ts_tensor_shape shape) {
    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    ts_u64 alloc = (ts_u64)shape.width * shape.height * shape.depth;
    return ts_tensor_create_alloc(arena, shape, alloc);
}
ts_tensor* ts_tensor_create_alloc(mg_arena* arena, ts_tensor_shape shape, ts_u64 alloc) {
    if (shape.width == 0) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot create ts_tensor of width 0");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }
    
    ts_u64 min_alloc = (ts_u64)shape.width * shape.height * shape.depth;
    if (alloc < min_alloc) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create ts_tensor, alloc is too small");

        return NULL;
    }

    ts_tensor* out = MGA_PUSH_STRUCT(arena, ts_tensor);

    _tensor_create_alloc_backend(arena, out, shape, alloc);
    
    return out;
}
void ts_tensor_destroy(ts_tensor* t) {
    _tensor_destroy_backend(t);
}
ts_tensor* ts_tensor_copy(mg_arena* arena, const ts_tensor* t, ts_b32 keep_alloc) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot copy NULL ts_tensor");

        return NULL;
    }

    ts_tensor_shape shape = t->shape;
    ts_u64 alloc = keep_alloc ? t->alloc : ((ts_u64)shape.width * shape.height * shape.depth);

    ts_tensor* out = ts_tensor_create_alloc(arena, shape, alloc);

    _tensor_copy_backend(out, t, out->alloc);

    return out;
}
ts_b32 ts_tensor_copy_ip(ts_tensor* out, const ts_tensor* t) {
    if (out == NULL || t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot copy ts_tensor: out and/or tensor is NULL");

        return false;
    }

    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot copy ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;
    if (out->data != t->data) {
        _tensor_copy_backend(out, t, size);
    }

    return true;
}

void ts_tensor_fill(ts_tensor* tensor, ts_f32 num) {
    if (tensor == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot fill NULL ts_tensor");
    }

    _tensor_fill_backend(tensor, num);
}

ts_tensor_index ts_tensor_argmax(const ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot get argmax of NULL ts_tensor");

        return (ts_tensor_index){ 0 };
    }

    return _tensor_argmax_backend(t);
}

ts_b32 ts_tensor_is_zero(const ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot test if NULL ts_tensor is zero");

        return false;
    }

    return _tensor_is_zero(t);
}
void ts_tensor_2d_view(ts_tensor* out, const ts_tensor* tensor, ts_u32 z) {
    if (out == NULL || tensor == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create 2d view will NULL ts_tensor(s)");

        return;
    }

    _tensor_2d_view_backend(out, tensor, z);
}

ts_b32 ts_tensor_dot_ip(ts_tensor* out, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot dot with NULL ts_tensor(s)");

        return false;
    }

    if (a->shape.depth != 1 || b->shape.depth != 1) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot dot ts_tensor in 3 dimensions");

        return false;
    }

    ts_tensor_shape a_shape = a->shape;
    ts_tensor_shape b_shape = b->shape;
    if (transpose_a) {
        ts_u32 tmp = a_shape.width;
        a_shape.width = a_shape.height;
        a_shape.height = tmp;
    }
    if (transpose_b) {
        ts_u32 tmp = b_shape.width;
        b_shape.width = b_shape.height;
        b_shape.height = tmp;
    }

    if (a_shape.width != b_shape.height) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot dot ts_tensor: shapes do not align");

        return false;
    }

    ts_tensor_shape out_shape = {
        .width = b_shape.width,
        .height = a_shape.height,
        .depth = 1
    };
    ts_u64 data_size = (ts_u64)out_shape.width * out_shape.height;

    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot dot ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = out_shape;

    // TODO: remove data copying

    ts_f32* a_data = a->data;
    ts_f32* b_data = b->data;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    if (a_data == out->data) {
        ts_u64 a_data_size = (ts_u64)a->shape.width * a->shape.height;
        a_data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, a_data_size);
        memcpy(a_data, a->data, sizeof(ts_f32) * a_data_size);
    }
    if (b_data == out->data) {
        ts_u64 b_data_size = (ts_u64)b->shape.width * b->shape.height;
        b_data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, b_data_size);
        memcpy(b_data, b->data, sizeof(ts_f32) * b_data_size);
    }

    ts_tensor real_a = {
        .alloc = a->alloc,
        .shape = a->shape,
        .data = a_data
    };
    ts_tensor real_b = {
        .alloc = b->alloc,
        .shape = b->shape,
        .data = b_data
    };

    _tensor_dot_backend(out, transpose_a, transpose_b, &real_a, &real_b);

   
    mga_scratch_release(scratch);

    return true;
}
ts_tensor* ts_tensor_dot(mg_arena* arena, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b) {
    if (a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot dot with NULL ts_tensor(s)");

        return NULL;
    }

    ts_tensor_shape shape = {
        transpose_b ? b->shape.height : b->shape.width,
        transpose_a ? a->shape.width : a->shape.height,
        1
    };

    ts_tensor* out = ts_tensor_create(arena, shape);

    ts_tensor_dot_ip(out, transpose_a, transpose_b, a, b);

    return out;
}

ts_tensor_shape ts_tensor_conv_shape(ts_tensor_shape in_shape, ts_tensor_shape kernel_shape, ts_u32 stride_x, ts_u32 stride_y) {
    ts_tensor_shape out_shape = { 0, 0, 1 };

    if (stride_x == 0 || stride_y == 0) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create conv shape with strides of zero");

        return out_shape;
    }

    out_shape.width = (in_shape.width - kernel_shape.width) / stride_x + 1;
    out_shape.height = (in_shape.height - kernel_shape.height) / stride_y + 1;

    return out_shape;
}
ts_b32 ts_tensor_im2col_ip(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding) {
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot im2col with NULL ts_tensor(s)");

        return false;
    }

    if (stride == 0) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert image to cols with stride of zero");

        return false;
    }
    if (out->data == input->data) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert image to cols when out and input overlap");

        return false;
    }

    // Number of kernels that fit in input on the x and y axes
    ts_u32 x_kernels = (input->shape.width + padding * 2 - kernel_size) / stride + 1;
    ts_u32 y_kernels = (input->shape.height + padding * 2 - kernel_size) / stride + 1;

    ts_tensor_shape shape = {
        x_kernels * y_kernels,
        input->shape.depth * kernel_size * kernel_size,
        1
    };

    ts_u64 out_alloc = (ts_u64)shape.width * shape.height * shape.depth;
    if (out->alloc < out_alloc) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot convert image to cols: not enough space in out");
        #endif

        return false;
    }

    out->shape = shape;

    _tensor_im2col_backend(out, input, kernel_size, stride, padding, x_kernels, y_kernels);

    return true;
}
ts_tensor* ts_tensor_im2col(mg_arena* arena, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert NULL ts_tensor to cols");

        return NULL;
    }

    if (stride == 0) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert image to cols with stride of zero");

        return false;
    }

    // Number of kernels that fit in input on the x and y axes
    ts_u32 x_kernels = (input->shape.width + padding * 2 - kernel_size) / stride + 1;
    ts_u32 y_kernels = (input->shape.height + padding * 2 - kernel_size) / stride + 1;

    ts_tensor_shape shape = {
        x_kernels * y_kernels,
        input->shape.depth * kernel_size * kernel_size,
        1
    };

    mga_temp maybe_temp = mga_temp_begin(arena);
    ts_tensor* out = ts_tensor_create(arena, shape);

    if (!ts_tensor_im2col_ip(out, input, kernel_size, stride, padding)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}

ts_b32 ts_tensor_col2im_ip(ts_tensor* out, const ts_tensor* input, ts_tensor_shape out_shape, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding) {
   if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot col2im with NULL ts_tensor(s)");

        return false;
    }

    if (stride == 0) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert cols to image with stride of zero");

        return false;
    }
    if (out->data == input->data) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert cols to image when out and input overlap");

        return false;
    } 

    ts_u64 out_alloc = (ts_u64)out_shape.width * out_shape.height * out_shape.depth;
    if (out->alloc < out_alloc) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot convert cols to image: not enough space in out");
        #endif

        return false;
    }
    out->shape = out_shape;

    ts_u32 x_kernels = (out_shape.width + padding * 2 - kernel_size) / stride + 1;
    ts_u32 y_kernels = (out_shape.height + padding * 2 - kernel_size) / stride + 1;

    _tensor_col2im_backend(out, input, kernel_size, stride, padding, x_kernels, y_kernels);

    return true;
}
ts_tensor* ts_tensor_col2im(mg_arena* arena, const ts_tensor* input, ts_tensor_shape out_shape, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding) {
    if (input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert NULL ts_tensor to image");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);
    ts_tensor* out = ts_tensor_create(arena, out_shape);

    if (!ts_tensor_col2im_ip(out, input, out_shape, kernel_size, stride, padding)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}

ts_b32 ts_tensor_transpose_ip(ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot transpose NULL ts_tensor");

        return false;
    }

    if (t->shape.depth != 1) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot transpose ts_tensor with depth");

        return false;
    }

    ts_u32 temp_width = t->shape.width;
    t->shape.width = t->shape.height;
    t->shape.height = temp_width;

    // If it is 1d, you do not need to move around the numbers
    if (t->shape.width == 1 || t->shape.height == 1) {
        return true;
    }

    // Creating temporary copy of data
    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_tensor* orig = ts_tensor_copy(scratch.arena, t, false);
    orig->shape.width = t->shape.height;
    orig->shape.height = t->shape.width;

    _tensor_transpose_backend(t, orig);

    ts_tensor_destroy(orig);

    mga_scratch_release(scratch);

    return true;
}
ts_tensor* ts_tensor_transpose(mg_arena* arena, const ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot transpose NULL ts_tensor");

        return NULL;
    }

    if (t->shape.depth != 1) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot transpose ts_tensor with depth");

        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, (ts_tensor_shape){ t->shape.height, t->shape.width, 1 });

    _tensor_transpose_backend(out, t);

    return out;
}

ts_b32 ts_tensor_add_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add NULL ts_tensor(s)");

        return false;
    }
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot add ts_tensor: shapes do not align");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot add ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_add_backend(out, a, b);
    
    return true;
}
ts_b32 ts_tensor_sub_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot subtract NULL ts_tensor(s)");

        return false;
    }
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot subtract ts_tensor: shapes do not align");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot subtract ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_sub_backend(out, a, b);
    
    return true;
}
ts_b32 ts_tensor_component_mul_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot component multiply NULL ts_tensor(s)");

        return false;
    }
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot multiply ts_tensor: shapes do not align");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot multiply ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_component_mul_backend(out, a, b);
    
    return true;
}
ts_b32 ts_tensor_component_div_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (out == NULL || a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot component divide NULL ts_tensor(s)");

        return false;
    }
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        TS_ERR(TS_ERR_BAD_SHAPE, "Cannot divide ts_tensor: shapes do not align");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot divide ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = a->shape;

    _tensor_component_div_backend(out, a, b);
    
    return true;
}
ts_b32 ts_tensor_add_all_ip(ts_tensor* out, const ts_tensor* t, ts_f32 x) {
    if (out == NULL || t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add all with NULL ts_tensor(s)");

        return false;
    }

    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot add all to ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_add_all_backend(out, t, x);

    return true;
}
ts_b32 ts_tensor_scale_ip(ts_tensor* out, const ts_tensor* t, ts_f32 s) {
    if (out == NULL || t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot scale NULL ts_tensor(s)");

        return false;
    }
    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot scale ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_scale_backend(out, t, s);

    return true;
}
ts_b32 ts_tensor_sqrt_ip(ts_tensor* out, const ts_tensor* t) {
    if (out == NULL || t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot sqrt NULL ts_tensor(s)");

        return false;
    }
    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_IP_ALLOC_ERRORS
        TS_ERR(TS_ERR_ALLOC_SIZE, "Cannot sqrt ts_tensor: not enough space in out");
        #endif

        return false;
    }

    out->shape = t->shape;

    _tensor_sqrt_backend(out, t);

    return true;
}

ts_tensor* ts_tensor_add(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    if (a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add NULL ts_tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_add_ip(out, a, b)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_sub(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    if (a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot sub NULL ts_tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_sub_ip(out, a, b)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_component_mul(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    if (a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot component multiply NULL ts_tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_component_mul_ip(out, a, b)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_component_div(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    if (a == NULL || b == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot component divide NULL ts_tensor(s)");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_component_div_ip(out, a, b)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_add_all(mg_arena* arena, const ts_tensor* t, ts_f32 x) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot add all to NULL ts_tensor");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, t->shape);

    if (!ts_tensor_add_all_ip(out, t, x)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;

}
ts_tensor* ts_tensor_scale(mg_arena* arena, const ts_tensor* t, ts_f32 s) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot scale NULL ts_tensor");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, t->shape);

    if (!ts_tensor_scale_ip(out, t, s)) {
        ts_tensor_destroy(out);
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_sqrt(mg_arena* arena, const ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot sqrt NULL ts_tensor");

        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, t->shape);

    ts_tensor_sqrt_ip(out, t);

    return out;
}

ts_f32* ts_tensor_copy_data(mg_arena* arena, const ts_tensor* t) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot copy data from NULL ts_tensor");

        return NULL;
    }


    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    ts_f32* out = MGA_PUSH_ARRAY(arena, ts_f32, size);

    _tensor_get_data_backend(out, t);

    return out;
}
void ts_tensor_get_data(ts_f32* out, const ts_tensor* t) {
    if (out == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot get ts_tensor data with NULL out pointer");

        return;
    }
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot get data of NULL ts_tensor");

        return;
    }

    _tensor_get_data_backend(out, t);
}
void ts_tensor_set_data(ts_tensor* t, ts_f32* data) {
    if (t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot set data of NULL ts_tensor");

        return;
    }
    if (data == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot set ts_tensor data with NULL data");

        return;
    }

    _tensor_set_data_backend(t, data);
}

void ts_tensor_list_push_existing(ts_tensor_list* list, ts_tensor* tensor, ts_string8 name, ts_tensor_node* node) {
    if (list == NULL || tensor == NULL || node == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot push node to tensor list: list, node, or tensor is NULL");

        return;
    }

    node->tensor = tensor;
    node->name = name;

    TS_SLL_PUSH_BACK(list->first, list->last, node);

    list->size++;
}
void ts_tensor_list_push(mg_arena* arena, ts_tensor_list* list, ts_tensor* tensor, ts_string8 name) {
    if (list == NULL || tensor == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot push tensor to list: list or tensor is NULL");

        return;
    }

    ts_tensor_node* node = MGA_PUSH_ZERO_STRUCT(arena, ts_tensor_node);
    ts_tensor_list_push_existing(list, tensor, name, node);
}
ts_tensor* ts_tensor_list_get(const ts_tensor_list* list, ts_string8 name) {
    if (list == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot get tensor from NULL list");

        return NULL;
    }
    
    ts_tensor* out = NULL;

    for (ts_tensor_node* node = list->first; node != NULL; node = node->next) {
        if (ts_str8_equals(node->name, name)) {
            return node->tensor;
        }
    }

    return out;
}

/*
TODO: Figure out how to make it endian independent

File Format (*.tst):
- Header "TS_tensors"
- ts_u32 num_tensors
- List of ts_tensors
    - Name
        - ts_u64 size
        - u8* str (of length size)
    - ts_tensor
        - ts_u32 width, height, depth
        - ts_f32* data (of length width*height*depth)
*/

static const ts_string8 _tst_header = {
    .size = 10,
    .str = (ts_u8*)"TS_tensors"
};

ts_string8 ts_tensor_get_tst_header(void) {
    return _tst_header;
}

#define _WRITE_DATA(size, data) do { \
        memcpy(str_buf_ptr, (data), (size)); \
        str_buf_ptr += (size); \
    } while (0)

ts_string8 ts_tensor_list_to_str(mg_arena* arena, const ts_tensor_list* list) {
    if (list == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot convert NULL tensor list to string");

        return (ts_string8){ 0 };
    }

    ts_u64 str_size = 0;
    ts_u8* str_buf = NULL;

    str_size += _tst_header.size;
    str_size += sizeof(ts_u32); // for number of ts_tensors

    for (ts_tensor_node* node = list->first; node != NULL; node = node->next) {
        str_size += sizeof(ts_u64); // for str size
        str_size += node->name.size; // for str

        ts_tensor_shape shape = node->tensor->shape;

        str_size += sizeof(ts_u32) * 3; // for width, height, and depth
        str_size += (ts_u64)shape.width * shape.height * shape.depth * sizeof(ts_f32); // for data
    }

    str_buf = MGA_PUSH_ARRAY(arena, ts_u8, str_size);
    ts_u8* str_buf_ptr = str_buf;

    _WRITE_DATA(_tst_header.size, _tst_header.str);
    _WRITE_DATA(sizeof(ts_u32), &list->size);

    for (ts_tensor_node* node = list->first; node != NULL; node = node->next) {
        _WRITE_DATA(sizeof(ts_u64), &node->name.size);
        _WRITE_DATA(node->name.size, node->name.str);

        ts_tensor_shape shape = node->tensor->shape;

        _WRITE_DATA(sizeof(ts_u32), &shape.width);
        _WRITE_DATA(sizeof(ts_u32), &shape.height);
        _WRITE_DATA(sizeof(ts_u32), &shape.depth);

        ts_u64 data_size = (ts_u64)shape.width * shape.height * shape.depth * sizeof(ts_f32);

        mga_temp scratch = mga_scratch_get(NULL, 0);
        ts_f32* data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, data_size);

        _tensor_set_data_backend(node->tensor, data);

        _WRITE_DATA(data_size, data);

        mga_scratch_release(scratch);
    }

    if (str_size != (ts_u64)(str_buf_ptr - str_buf)) {
        TS_ERR(TS_ERR_GENERAL, "Cannnot create ts_tensor string: buffer was not filled");

        return (ts_string8){ 0 };
    }

    return (ts_string8){ .str = str_buf, .size = str_size };
}

#define _READ_DATA(data_size, data) do { \
        if (pos + (data_size) > str.size) { \
            memset((data), 0, (data_size)); \
        } else { \
            memcpy((data), &str.str[pos], (data_size)); \
            pos += (data_size); \
        } \
    } while (0)

ts_tensor_list ts_tensor_list_from_str(mg_arena* arena, ts_string8 str) {
    if (!ts_str8_equals(_tst_header, ts_str8_substr(str, 0, _tst_header.size))) {
        TS_ERR(TS_ERR_PARSE, "Cannot read ts_tensor string: ts_tensor header not found");
        
        return (ts_tensor_list){ 0 };
    }

    ts_u64 pos = _tst_header.size;

    ts_tensor_list out = { 0 };
    
    ts_u32 size = 0;
    _READ_DATA(sizeof(ts_u32), &size);

    for (ts_u32 i = 0; i < size; i++) {
        ts_u64 name_size = 0;
        _READ_DATA(sizeof(ts_u64), &name_size);

        ts_string8 name = {
            .size = name_size,
            .str = MGA_PUSH_ZERO_ARRAY(arena, ts_u8, name_size)
        };

        _READ_DATA(name_size, name.str);

        ts_u32 width = 0;
        ts_u32 height = 0;
        ts_u32 depth = 0;

        _READ_DATA(sizeof(ts_u32), &width);
        _READ_DATA(sizeof(ts_u32), &height);
        _READ_DATA(sizeof(ts_u32), &depth);

        ts_tensor* tensor = ts_tensor_create(arena, (ts_tensor_shape){ width, height, depth });
        ts_u64 data_size = (ts_u64)width * height * depth * sizeof(ts_f32);

        mga_temp scratch = mga_scratch_get(NULL, 0);

        ts_f32* data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, data_size);
        _READ_DATA(data_size, data);
        _tensor_set_data_backend(tensor, data);

        mga_scratch_release(scratch);

        ts_tensor_list_push(arena, &out, tensor, name);
    }

    if (pos > str.size) {
        TS_ERR(TS_ERR_PARSE, "Could not load all ts_tensors: cannot read outisde string bounds");
    }

    return out;
}

void ts_tensor_list_save(const ts_tensor_list* list, ts_string8 file_name) {
    if (list == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot save NULL list");

        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 file_str = ts_tensor_list_to_str(scratch.arena, list);

    if (file_str.size == 0) {
        TS_ERR(TS_ERR_GENERAL, "Cannnot write ts_tensor file: string was not created");
    } else {
        ts_string8_list output_list = { 0 };
        ts_str8_list_push(scratch.arena, &output_list, file_str);

        ts_file_write(file_name, output_list);
    }

    mga_scratch_release(scratch);
}
   
ts_tensor_list ts_tensor_list_load(mg_arena* arena, ts_string8 file_name) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 file = ts_file_read(scratch.arena, file_name);
    if (file.size == 0) {
        TS_ERR(TS_ERR_IO, "Cannot load ts_tensors: failed to read file");
        
        mga_scratch_release(scratch);
        return (ts_tensor_list){ 0 };
    }

    ts_tensor_list out = ts_tensor_list_from_str(arena, file);

    mga_scratch_release(scratch);

    return out;
}
