#include "tensor.h"

#include "os.h"

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
        fprintf(stderr, "Unable to create ts_tensor of width 0\n");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }
    
    ts_u64 min_alloc = (ts_u64)shape.width * shape.height * shape.depth;
    if (alloc < min_alloc) {
        fprintf(stderr, "Cannot create ts_tensor, alloc is too small\n");

        return NULL;
    }

    ts_tensor* out = MGA_PUSH_STRUCT(arena, ts_tensor);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, ts_f32, alloc);
    
    return out;
}
ts_tensor* ts_tensor_copy(mg_arena* arena, const ts_tensor* t, ts_b32 keep_alloc) {
    ts_tensor_shape shape = t->shape;
    ts_u64 alloc = keep_alloc ? t->alloc : ((ts_u64)shape.width * shape.height * shape.depth);

    ts_tensor* out = MGA_PUSH_STRUCT(arena, ts_tensor);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, ts_f32, out->alloc);

    memcpy(out->data, t->data, sizeof(ts_f32) * out->alloc);

    return out;
}
ts_b32 ts_tensor_copy_ip(ts_tensor* out, const ts_tensor* t) {
    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot copy ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    memcpy(out->data, t->data, size * sizeof(ts_f32));

    return true;
}

void ts_tensor_fill(ts_tensor* ts_tensor, ts_f32 num) {
    ts_tensor_shape shape = ts_tensor->shape;
    ts_u64 size = (ts_u64)shape.width * shape.height * shape.depth;

    for (ts_u64 i = 0; i < size; i++) {
        ts_tensor->data[i] = num;
    }
}

ts_tensor_index ts_tensor_argmax(const ts_tensor* t) {
    ts_f32 max_num = t->data[0];
    ts_tensor_index max_index = { 0, 0, 0 };

    for (ts_u64 z = 0; z < t->shape.depth; z++) {
        for (ts_u64 y = 0; y < t->shape.height; y++) {
            for (ts_u64 x = 0; x < t->shape.width; x++) {
                if (t->data[x + y * t->shape.width + z * t->shape.width * t->shape.height] > max_num) {
                    max_num = t->data[x + y * t->shape.width + z * t->shape.width * t->shape.height];
                    max_index = (ts_tensor_index){ x, y, z };
                }
            }
        }
    }

    return max_index;
}

ts_b32 ts_tensor_is_zero(const ts_tensor* t) {
    ts_b32 is_zero = true;

    ts_u64 size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    for (ts_u64 i = 0; i < size; i++) {
        if (t->data[i] != 0.0f) {
            is_zero = false;
            break;
        }
    }

    return is_zero;
}

ts_tensor* ts_tensor_slice(mg_arena* arena, const ts_tensor* t, ts_tensor_index start, ts_tensor_index end) {
    if (end.x > t->shape.width || end.y > t->shape.height || end.z > t->shape.depth) {
        fprintf(stderr, "Cannot create slice past end of ts_tensor\n");

        return NULL;
    }
    if (start.x > end.x || start.y > end.y || start.z > end.z) {
        fprintf(stderr, "Start of ts_tensor slice cannot exceed end\n");

        return NULL;
    }
    
    ts_tensor_shape slice_shape = {
        .width = end.x - start.x,
        .height = end.y - start.y,
        .depth = end.z - start.z,
    };

    if (slice_shape.width > t->shape.width || slice_shape.height > t->shape.height || slice_shape.depth > t->shape.depth) {
        fprintf(stderr, "Cannot create slice greater than original ts_tensor\n");
        
        return NULL;
    }

    ts_tensor* slice = ts_tensor_create(arena, slice_shape);
    
    if (slice->shape.depth == 1) { // Fast path for 2d slice of 3d ts_tensor
        ts_u64 start_i = (ts_u64)start.z * t->shape.width * t->shape.height;

        if (slice->shape.width == t->shape.width  && slice->shape.height == t->shape.height) {
            ts_u64 slice_size = (ts_u64)t->shape.width * t->shape.height;

            memcpy(slice->data, &t->data[start_i], sizeof(ts_f32) * slice_size);
        } else {
            for (ts_u64 y = start.y; y < end.y; y++) {
                for (ts_u64 x = start.x; x < end.x; x++) {
                    slice->data[x + y * slice->shape.height] = t->data[start_i + x + y * t->shape.height];
                }
            }
        }
    } else { // General case
        for (ts_u64 z = start.z; z < end.z; z++) {
            for (ts_u64 y = start.y; y < end.y; y++) {
                for (ts_u64 x = start.x; x < end.x; x++) {
                    ts_u64 slice_i = x + y * slice->shape.width + z * slice->shape.width * slice->shape.height;
                    ts_u64 ts_tensor_i = x + y * t->shape.width + z * t->shape.width * t->shape.height;
                    
                    slice->data[slice_i] = t->data[ts_tensor_i];
                }
            }
        }
    }

    return slice;
}
ts_tensor* ts_tensor_slice_size(mg_arena* arena, const ts_tensor* ts_tensor, ts_tensor_index start, ts_tensor_shape shape) {
    ts_tensor_index end = {
        start.x + shape.width,
        start.y + shape.height,
        start.z + shape.depth
    };

    return ts_tensor_slice(arena, ts_tensor, start, end);
}
void ts_tensor_2d_view(ts_tensor* out, const ts_tensor* ts_tensor, ts_u32 z) {
    out->shape = (ts_tensor_shape) {
        .width = ts_tensor->shape.width,
        .height = ts_tensor->shape.height,
        .depth = 1
    };
    out->alloc = (ts_u64)out->shape.width * out->shape.height;

    ts_u64 start_i = (ts_u64)z * ts_tensor->shape.width * ts_tensor->shape.height;

    out->data = &ts_tensor->data[start_i];
}

ts_b32 ts_tensor_dot_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (a->shape.depth != 1 || b->shape.depth != 1) {
        fprintf(stderr, "Cannot dot ts_tensor in 3 dimensions\n");

        return false;
    }
    if (a->shape.width != b->shape.height) {
        fprintf(stderr, "Cannot dot ts_tensor: shapes do not align\n");

        return false;
    }

    ts_tensor_shape shape = {
        .width = b->shape.width,
        .height = a->shape.height,
        .depth = 1
    };
    ts_u64 data_size = (ts_u64)shape.width * shape.height;

    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot dot ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    ts_u32 a_width = a->shape.width;
    ts_u32 b_width = b->shape.width;
    
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

    out->shape = shape;
    memset(out->data, 0, sizeof(ts_f32) * data_size);

    for (ts_u32 y = 0; y < shape.height; y++) {
        for (ts_u32 i = 0; i < a_width; i++) {
            // This does not change throughout the inner loop
            ts_f32 a_elem = a_data[(ts_u64)i + (ts_u64)y * a_width];
            for (ts_u32 x = 0; x < shape.width; x++) {
                out->data[(ts_u64)x + (ts_u64)y * shape.width] += a_elem * b_data[(ts_u64)x + (ts_u64)i * b_width];
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}
ts_tensor* ts_tensor_dot(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    ts_tensor_shape shape = {
        b->shape.width,
        a->shape.height,
        1
    };

    ts_tensor* out = ts_tensor_create(arena, shape);

    ts_tensor_dot_ip(out, a, b);

    return out;
}

ts_tensor_shape ts_tensor_conv_shape(ts_tensor_shape in_shape, ts_tensor_shape kernel_shape, ts_u32 stride_x, ts_u32 stride_y) {
    ts_tensor_shape out_shape = { 0, 0, 1 };

    if (stride_x == 0 || stride_y == 0) {
        fprintf(stderr, "Cannot create conv shape: strides cannot be zero\n");

        return out_shape;
    }

    out_shape.width = (in_shape.width - kernel_shape.width + 1) / stride_x;
    out_shape.height = (in_shape.height - kernel_shape.height + 1) / stride_y;

    return out_shape;
}
ts_b32 ts_tensor_conv_ip(ts_tensor* out, const ts_tensor* input, const ts_tensor* kernel, ts_u32 stride_x, ts_u32 stride_y) {
    if (stride_x == 0 || stride_y == 0) {
        fprintf(stderr, "Cannot conv tensors: strides cannot be zero\n");

        return false;
    }
    ts_tensor_shape out_shape = ts_tensor_conv_shape(input->shape, kernel->shape, stride_x, stride_y);

    ts_u64 out_alloc = (ts_u64)out_shape.width * out_shape.height * out_shape.depth;

    if (out->alloc < out_alloc) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot add ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Checking for shared data
    if (out->data == input->data) {
        input = ts_tensor_copy(scratch.arena, input, false);
    }
    if (out->data == kernel->data) {
        kernel = ts_tensor_copy(scratch.arena, kernel, false);
    }

    out->shape = out_shape;

    // Input pos: i_x, i_y
    // Output pos: o_x, o_y
    // Kernel pos: k_x, k_y
    for (ts_u32 o_y = 0, i_y = 0; o_y < out->shape.height; o_y++, i_y += stride_y) {
        for (ts_u32 o_x = 0, i_x = 0; o_x < out->shape.width; o_x++, i_x += stride_x) {
            ts_u64 out_pos = (ts_u64)o_x + (ts_u64)o_y * out->shape.width;

            out->data[out_pos] = 0.0f;

            for (ts_u32 k_y = 0; k_y < kernel->shape.height; k_y++) {
                for (ts_u32 k_x = 0; k_x < kernel->shape.width; k_x++) {
                    ts_u64 in_pos = (ts_u64)(i_x + k_x) + (ts_u64)(i_y + k_y) * input->shape.width;
                    ts_u64 kernel_pos = (ts_u64)k_x + (ts_u64)k_y * kernel->shape.width;

                    out->data[out_pos] += input->data[in_pos] * kernel->data[kernel_pos];
                }
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}
ts_tensor* ts_tensor_conv(mg_arena* arena, const ts_tensor* input, const ts_tensor* kernel, ts_u32 stride_x, ts_u32 stride_y) {
    ts_tensor_shape out_shape = ts_tensor_conv_shape(input->shape, kernel->shape, stride_x, stride_y);

    ts_tensor* out = ts_tensor_create(arena, out_shape);

    ts_tensor_conv_ip(out, input, kernel, stride_x, stride_y);

    return out;
}

void ts_tensor_transpose_ip(ts_tensor* t) {
    if (t->shape.depth != 1) {
        fprintf(stderr, "Cannot transpose ts_tensor with depth\n");

        return;
    }

    ts_u32 temp_width = t->shape.width;
    t->shape.width = t->shape.height;
    t->shape.height = temp_width;

    // If it is 1d, you do not need to move around the numbers
    if (t->shape.width == 1 || t->shape.height == 1) {
        return;
    }

    // Creating temporary copy of data
    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_u32 orig_width = t->shape.height;

    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height; // depth == 1
    ts_f32* orig_data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, data_size);
    memcpy(orig_data, t->data, sizeof(ts_f32) * data_size);

    for (ts_u64 x = 0; x < t->shape.width; x++) {
        for (ts_u64 y = 0; y < t->shape.height; y++) {
            t->data[x + y * t->shape.width] = orig_data[y + x * orig_width];
        }
    }

    mga_scratch_release(scratch);
}
ts_tensor* ts_tensor_transpose(mg_arena* arena, const ts_tensor* t) {
    if (t->shape.depth != 1) {
        fprintf(stderr, "Cannot transpose ts_tensor with depth\n");

        return NULL;
    }

    ts_tensor* out = ts_tensor_create(arena, (ts_tensor_shape){ t->shape.height, t->shape.width, 1 });

    for (ts_u64 x = 0; x < t->shape.width; x++) {
        for (ts_u64 y = 0; y < t->shape.height; y++) {
            out->data[x + y * out->shape.width] = t->data[y + x * t->shape.width];
        }
    }

    return out;
}

ts_b32 ts_tensor_add_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot add ts_tensor: shapes do not align\n");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot add ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    return true;
}
ts_b32 ts_tensor_sub_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot subtract ts_tensor: shapes do not align\n");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot subtract ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    
    return true;
}
ts_b32 ts_tensor_component_mul_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot multiply ts_tensor: shapes do not align\n");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot multiply ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    return true;
}
ts_b32 ts_tensor_component_div_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b) {
    if (!ts_tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot divide ts_tensor: shapes do not align\n");

        return false;
    }
    
    ts_u64 data_size = (ts_u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot divide ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }
    
    return true;
}
ts_b32 ts_tensor_scale_ip(ts_tensor* out, const ts_tensor* t, ts_f32 s) {
    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot scale ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = t->data[i] * s;
    }

    return true;
}
ts_b32 ts_tensor_sqrt_ip(ts_tensor* out, const ts_tensor* t) {
    ts_u64 data_size = (ts_u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TS_TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot sqrt ts_tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    for (ts_u64 i = 0; i < data_size; i++) {
        out->data[i] = sqrtf(t->data[i]);
    }

    return true;
}

ts_tensor* ts_tensor_add(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_add_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_sub(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_sub_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_component_mul(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_component_mul_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_component_div(mg_arena* arena, const ts_tensor* a, const ts_tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, a->shape);

    if (!ts_tensor_component_div_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_scale(mg_arena* arena, const ts_tensor* t, ts_f32 s) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_tensor* out = ts_tensor_create(arena, t->shape);

    if (!ts_tensor_scale_ip(out, t, s)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
ts_tensor* ts_tensor_sqrt(mg_arena* arena, const ts_tensor* t) {
    ts_tensor* out = ts_tensor_create(arena, t->shape);

    ts_tensor_sqrt_ip(out, t);

    return out;
}

void ts_tensor_list_push_existing(ts_tensor_list* list, ts_tensor* ts_tensor, ts_string8 name, ts_tensor_node* node) {
    node->tensor = ts_tensor;
    node->name = name;

    TS_SLL_PUSH_BACK(list->first, list->last, node);

    list->size++;
}
void ts_tensor_list_push(mg_arena* arena, ts_tensor_list* list, ts_tensor* ts_tensor, ts_string8 name) {
    ts_tensor_node* node = MGA_PUSH_ZERO_STRUCT(arena, ts_tensor_node);
    ts_tensor_list_push_existing(list, ts_tensor, name, node);
}
ts_tensor* ts_tensor_list_get(const ts_tensor_list* list, ts_string8 name) {
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

File Format (*.tpt):
- Header "TP_tensors"
- ts_u32 num_tensors
- List of ts_tensors
    - Name
        - ts_u64 size
        - u8* str (of length size)
    - ts_tensor
        - ts_u32 width, height, depth
        - ts_f32* data (of length width*height*depth)
*/

static const ts_string8 _tpt_header = {
    .size = 10,
    .str = (ts_u8*)"TP_tensors"
};

ts_string8 ts_tensor_get_tpt_header(void) {
    return _tpt_header;
}

#define _WRITE_DATA(size, data) do { \
        memcpy(str_buf_ptr, (data), (size)); \
        str_buf_ptr += (size); \
    } while (0)

ts_string8 ts_tensor_list_to_str(mg_arena* arena, const ts_tensor_list* list) {
    ts_u64 str_size = 0;
    ts_u8* str_buf = NULL;

    str_size += _tpt_header.size;
    str_size += sizeof(ts_u32); // for number of ts_tensors

    for (ts_tensor_node* node = list->first; node != NULL; node = node->next) {
        str_size += sizeof(ts_u64); // for str size
        str_size += node->name.size; // for str

        ts_tensor_shape shape = node->tensor->shape;

        str_size += sizeof(ts_u32) * 3; // for width, height, and depth
        str_size += (ts_u64)shape.width * shape.height * shape.depth * sizeof(ts_f32); // for data
    }

    str_buf = MGA_PUSH_ARRAY(arena, ts_u8, str_size);
    if (str_buf == NULL) {
        fprintf(stderr, "Cannot create ts_tensor string: failed to create buffer on arena\n");

        return (ts_string8){ 0 };
    }
    ts_u8* str_buf_ptr = str_buf;

    _WRITE_DATA(_tpt_header.size, _tpt_header.str);
    _WRITE_DATA(sizeof(ts_u32), &list->size);

    for (ts_tensor_node* node = list->first; node != NULL; node = node->next) {
        _WRITE_DATA(sizeof(ts_u64), &node->name.size);
        _WRITE_DATA(node->name.size, node->name.str);

        ts_tensor_shape shape = node->tensor->shape;

        _WRITE_DATA(sizeof(ts_u32), &shape.width);
        _WRITE_DATA(sizeof(ts_u32), &shape.height);
        _WRITE_DATA(sizeof(ts_u32), &shape.depth);

        ts_u64 data_size = (ts_u64)shape.width * shape.height * shape.depth * sizeof(ts_f32);

        _WRITE_DATA(data_size, node->tensor->data);
    }

    if (str_size != (ts_u64)(str_buf_ptr - str_buf)) {
        fprintf(stderr, "Cannnot create ts_tensor string: buffer was not filled\n");

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
    if (!ts_str8_equals(_tpt_header, ts_str8_substr(str, 0, _tpt_header.size))) {
        fprintf(stderr, "Cannot read ts_tensor string: ts_tensor header not found\n");
        
        return (ts_tensor_list){ 0 };
    }

    ts_u64 pos = _tpt_header.size;

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

        ts_tensor* ts_tensor = ts_tensor_create(arena, (ts_tensor_shape){ width, height, depth });
        ts_u64 data_size = (ts_u64)width * height * depth * sizeof(ts_f32);
        _READ_DATA(data_size, ts_tensor->data);

        ts_tensor_list_push(arena, &out, ts_tensor, name);
    }

    if (pos > str.size) {
        fprintf(stderr, "Could not load all ts_tensors: cannot read outisde string bounds\n");
    }

    return out;
}

void ts_tensor_list_save(const ts_tensor_list* list, ts_string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 file_str = ts_tensor_list_to_str(scratch.arena, list);

    if (file_str.size == 0) {
        fprintf(stderr, "Cannnot write ts_tensor file: string was not created\n");
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
        fprintf(stderr, "Cannot load ts_tensors: failed to read file\n");
        
        mga_scratch_release(scratch);
        return (ts_tensor_list){ 0 };
    }

    ts_tensor_list out = ts_tensor_list_from_str(arena, file);

    mga_scratch_release(scratch);

    return out;
}
