#include "tensor.h"

#include "os/os.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

b32 tensor_shape_eq(tensor_shape a, tensor_shape b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

tensor* tensor_create(mg_arena* arena, tensor_shape shape) {
    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    u64 alloc = (u64)shape.width * shape.height * shape.depth;
    return tensor_create_alloc(arena, shape, alloc);
}
tensor* tensor_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc) {
    if (shape.width == 0) {
        fprintf(stderr, "Unable to create tensor of width 0\n");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }
    
    u64 min_alloc = (u64)shape.width * shape.height * shape.depth;
    if (alloc < min_alloc) {
        fprintf(stderr, "Cannot create tensor, alloc is too small\n");

        return NULL;
    }

    tensor* out = MGA_PUSH_STRUCT(arena, tensor);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, alloc);
    
    return out;
}
tensor* tensor_copy(mg_arena* arena, const tensor* t, b32 keep_alloc) {
    tensor_shape shape = t->shape;
    u64 alloc = keep_alloc ? t->alloc : ((u64)shape.width * shape.height * shape.depth);

    tensor* out = MGA_PUSH_STRUCT(arena, tensor);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, out->alloc);

    memcpy(out->data, t->data, sizeof(f32) * out->alloc);

    return out;
}
b32 tensor_copy_ip(tensor* out, const tensor* t) {
    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot copy tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    memcpy(out->data, t->data, size * sizeof(f32));

    return true;
}

void tensor_fill(tensor* tensor, f32 num) {
    tensor_shape shape = tensor->shape;
    u64 size = (u64)shape.width * shape.height * shape.depth;

    for (u64 i = 0; i < size; i++) {
        tensor->data[i] = num;
    }
}

tensor_index tensor_argmax(const tensor* t) {
    f32 max_num = t->data[0];
    tensor_index max_index = { 0, 0, 0 };

    for (u64 z = 0; z < t->shape.depth; z++) {
        for (u64 y = 0; y < t->shape.height; y++) {
            for (u64 x = 0; x < t->shape.width; x++) {
                if (t->data[x + y * t->shape.width + z * t->shape.width * t->shape.height] > max_num) {
                    max_num = t->data[x + y * t->shape.width + z * t->shape.width * t->shape.height];
                    max_index = (tensor_index){ x, y, z };
                }
            }
        }
    }

    return max_index;
}

b32 tensor_is_zero(const tensor* t) {
    b32 is_zero = true;

    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    for (u64 i = 0; i < size; i++) {
        if (t->data[i] != 0.0f) {
            is_zero = false;
            break;
        }
    }

    return is_zero;
}

tensor* tensor_slice(mg_arena* arena, const tensor* t, tensor_index start, tensor_index end) {
    if (end.x > t->shape.width || end.y > t->shape.height || end.z > t->shape.depth) {
        fprintf(stderr, "Cannot create slice past end of tensor\n");

        return NULL;
    }
    if (start.x > end.x || start.y > end.y || start.z > end.z) {
        fprintf(stderr, "Start of tensor slice cannot exceed end\n");

        return NULL;
    }
    
    tensor_shape slice_shape = {
        .width = end.x - start.x,
        .height = end.y - start.y,
        .depth = end.z - start.z,
    };

    if (slice_shape.width > t->shape.width || slice_shape.height > t->shape.height || slice_shape.depth > t->shape.depth) {
        fprintf(stderr, "Cannot create slice greater than original tensor\n");
        
        return NULL;
    }

    tensor* slice = tensor_create(arena, slice_shape);
    
    if (slice->shape.depth == 1) { // Fast path for 2d slice of 3d tensor
        u64 start_i = (u64)start.z * t->shape.width * t->shape.height;

        if (slice->shape.width == t->shape.width  && slice->shape.height == t->shape.height) {
            u64 slice_size = (u64)t->shape.width * t->shape.height;

            memcpy(slice->data, &t->data[start_i], sizeof(f32) * slice_size);
        } else {
            for (u64 y = start.y; y < end.y; y++) {
                for (u64 x = start.x; x < end.x; x++) {
                    slice->data[x + y * slice->shape.height] = t->data[start_i + x + y * t->shape.height];
                }
            }
        }
    } else { // General case
        for (u64 z = start.z; z < end.z; z++) {
            for (u64 y = start.y; y < end.y; y++) {
                for (u64 x = start.x; x < end.x; x++) {
                    u64 slice_i = x + y * slice->shape.width + z * slice->shape.width * slice->shape.height;
                    u64 tensor_i = x + y * t->shape.width + z * t->shape.width * t->shape.height;
                    
                    slice->data[slice_i] = t->data[tensor_i];
                }
            }
        }
    }

    return slice;
}
tensor* tensor_slice_size(mg_arena* arena, const tensor* tensor, tensor_index start, tensor_shape shape) {
    tensor_index end = {
        start.x + shape.width,
        start.y + shape.height,
        start.z + shape.depth
    };

    return tensor_slice(arena, tensor, start, end);
}
void tensor_2d_view(tensor* out, const tensor* tensor, u32 z) {
    out->shape = (tensor_shape) {
        .width = tensor->shape.width,
        .height = tensor->shape.height,
        .depth = 1
    };
    out->alloc = (u64)out->shape.width * out->shape.height;

    u64 start_i = (u64)z * tensor->shape.width * tensor->shape.height;

    out->data = &tensor->data[start_i];
}

b32 tensor_dot_ip(tensor* out, const tensor* a, const tensor* b) {
    if (a->shape.depth != 1 || b->shape.depth != 1) {
        fprintf(stderr, "Cannot dot tensor in 3 dimensions\n");

        return false;
    }
    if (a->shape.width != b->shape.height) {
        fprintf(stderr, "Cannot dot tensor: shapes do not align\n");

        return false;
    }

    tensor_shape shape = {
        .width = b->shape.width,
        .height = a->shape.height,
        .depth = 1
    };
    u64 data_size = (u64)shape.width * shape.height;

    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot dot tensor: not enough space in out\n");
        #endif

        return false;
    }

    u32 a_width = a->shape.width;
    u32 b_width = b->shape.width;
    
    f32* a_data = a->data;
    f32* b_data = b->data;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    if (a_data == out->data) {
        u64 a_data_size = (u64)a->shape.width * a->shape.height;
        a_data = MGA_PUSH_ARRAY(scratch.arena, f32, a_data_size);
        memcpy(a_data, a->data, sizeof(f32) * a_data_size);
    }
    if (b_data == out->data) {
        u64 b_data_size = (u64)b->shape.width * b->shape.height;
        b_data = MGA_PUSH_ARRAY(scratch.arena, f32, b_data_size);
        memcpy(b_data, b->data, sizeof(f32) * b_data_size);
    }

    out->shape = shape;
    memset(out->data, 0, sizeof(f32) * data_size);

    for (u32 y = 0; y < shape.height; y++) {
        for (u32 i = 0; i < a_width; i++) {
            // This does not change throughout the inner loop
            f32 a_elem = a_data[(u64)i + (u64)y * a_width];
            for (u32 x = 0; x < shape.width; x++) {
                out->data[(u64)x + (u64)y * shape.width] += a_elem * b_data[(u64)x + (u64)i * b_width];
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}
tensor* tensor_dot(mg_arena* arena, const tensor* a, const tensor* b) {
    tensor_shape shape = {
        b->shape.width,
        a->shape.height,
        1
    };

    tensor* out = tensor_create(arena, shape);

    tensor_dot_ip(out, a, b);

    return out;
}

tensor_shape tensor_conv_shape(tensor_shape in_shape, tensor_shape kernel_shape, u32 stride_x, u32 stride_y) {
    tensor_shape out_shape = { 0, 0, 1 };

    if (stride_x == 0 || stride_y == 0) {
        fprintf(stderr, "Cannot create conv shape: strides cannot be zero\n");

        return out_shape;
    }

    out_shape.width = (in_shape.width - kernel_shape.width + 1) / stride_x;
    out_shape.height = (in_shape.height - kernel_shape.height + 1) / stride_y;

    return out_shape;
}
b32 tensor_conv_ip(tensor* out, const tensor* input, const tensor* kernel, u32 stride_x, u32 stride_y) {
    tensor_shape out_shape = tensor_conv_shape(input->shape, kernel->shape, stride_x, stride_y);

    u64 out_alloc = (u64)out_shape.width * out_shape.height * out_shape.depth;

    if (out->alloc < out_alloc) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot add tensor: not enough space in out\n");
        #endif

        return false;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Checking for shared data
    if (out->data == input->data) {
        input = tensor_copy(scratch.arena, input, false);
    }
    if (out->data == kernel->data) {
        kernel = tensor_copy(scratch.arena, kernel, false);
    }

    out->shape = out_shape;

    // Input pos: i_x, i_y
    // Output pos: o_x, o_y
    // Kernel pos: k_x, k_y
    for (u32 o_y = 0, i_y = 0; o_y < out->shape.height; o_y++, i_y += stride_y) {
        for (u32 o_x = 0, i_x = 0; o_x < out->shape.width; o_x++, i_x += stride_x) {
            u64 out_pos = (u64)o_x + (u64)o_y * out->shape.width;

            out->data[out_pos] = 0.0f;

            for (u32 k_y = 0; k_y < kernel->shape.height; k_y++) {
                for (u32 k_x = 0; k_x < kernel->shape.width; k_x++) {
                    u64 in_pos = (u64)(i_x + k_x) + (u64)(i_y + k_y) * input->shape.width;
                    u64 kernel_pos = (u64)k_x + (u64)k_y * kernel->shape.width;

                    out->data[out_pos] += input->data[in_pos] * kernel->data[kernel_pos];
                }
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}
tensor* tensor_conv(mg_arena* arena, const tensor* input, const tensor* kernel, u32 stride_x, u32 stride_y) {
    tensor_shape out_shape = tensor_conv_shape(input->shape, kernel->shape, stride_x, stride_y);

    tensor* out = tensor_create(arena, out_shape);

    tensor_conv_ip(out, input, kernel, stride_x, stride_y);

    return out;
}

void tensor_transpose(tensor* t) {
    if (t->shape.depth != 1) {
        fprintf(stderr, "Cannot transpose tensor with depth");

        return;
    }

    u32 temp_width = t->shape.width;
    t->shape.width = t->shape.height;
    t->shape.height = temp_width;

    // If it is 1d, you do not need to move around the numbers
    if (t->shape.width == 1 || t->shape.height == 1) {
        return;
    }

    // Creating temporary copy of data
    mga_temp scratch = mga_scratch_get(NULL, 0);

    u32 orig_width = t->shape.height;

    u64 data_size = (u64)t->shape.width * t->shape.height; // depth == 1
    f32* orig_data = MGA_PUSH_ARRAY(scratch.arena, f32, data_size);
    memcpy(orig_data, t->data, sizeof(f32) * data_size);

    for (u64 y = 0; y < t->shape.height; y++) {
        for (u64 x = 0; x < t->shape.width; x++) {
            t->data[x + y * t->shape.width] = orig_data[y + x * orig_width];
        }
    }

    mga_scratch_release(scratch);
}

b32 tensor_add_ip(tensor* out, const tensor* a, const tensor* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot add tensor: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot add tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    return true;
}
b32 tensor_sub_ip(tensor* out, const tensor* a, const tensor* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot subtract tensor: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot subtract tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    
    return true;
}
b32 tensor_component_mul_ip(tensor* out, const tensor* a, const tensor* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot multiply tensor: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot multiply tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    return true;
}
b32 tensor_component_div_ip(tensor* out, const tensor* a, const tensor* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot divide tensor: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot divide tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }
    
    return true;
}
b32 tensor_scale_ip(tensor* out, const tensor* t, f32 s) {
    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot scale tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = t->data[i] * s;
    }

    return true;
}
b32 tensor_sqrt_ip(tensor* out, const tensor* t) {
    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot sqrt tensor: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = sqrtf(t->data[i]);
    }

    return true;
}

tensor* tensor_add(mg_arena* arena, const tensor* a, const tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_add_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensor* tensor_sub(mg_arena* arena, const tensor* a, const tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_sub_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensor* tensor_component_mul(mg_arena* arena, const tensor* a, const tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_component_mul_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensor* tensor_component_div(mg_arena* arena, const tensor* a, const tensor* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, a->shape);

    if (!tensor_component_div_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensor* tensor_scale(mg_arena* arena, const tensor* t, f32 s) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensor* out = tensor_create(arena, t->shape);

    if (!tensor_scale_ip(out, t, s)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensor* tensor_sqrt(mg_arena* arena, const tensor* t) {
    tensor* out = tensor_create(arena, t->shape);

    tensor_sqrt_ip(out, t);

    return out;
}

void tensor_list_push_existing(tensor_list* list, tensor* tensor, string8 name, tensor_node* node) {
    node->tensor = tensor;
    node->name = name;

    SLL_PUSH_BACK(list->first, list->last, node);

    list->size++;
}
void tensor_list_push(mg_arena* arena, tensor_list* list, tensor* tensor, string8 name) {
    tensor_node* node = MGA_PUSH_ZERO_STRUCT(arena, tensor_node);
    tensor_list_push_existing(list, tensor, name, node);
}
tensor* tensor_list_get(const tensor_list* list, string8 name) {
    tensor* out = NULL;

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        if (str8_equals(node->name, name)) {
            return node->tensor;
        }
    }

    return out;
}

/*
TODO: Figure out how to make it endian independent

File Format (*.tpt):
- Header "TP_tensors"
- u32 num_tensors
- List of tensors
    - Name
        - u64 size
        - u8* str (of length size)
    - Tensor
        - u32 width, height, depth
        - f32* data (of length width*height*depth)
*/

static const string8 _tpt_header = {
    .size = 10,
    .str = (u8*)"TP_tensors"
};

string8 tensor_get_tpt_header(void) {
    return _tpt_header;
}

#define _WRITE_DATA(size, data) do { \
        memcpy(str_buf_ptr, (data), (size)); \
        str_buf_ptr += (size); \
    } while (0)

string8 tensor_list_to_str(mg_arena* arena, const tensor_list* list) {
    u64 str_size = 0;
    u8* str_buf = NULL;

    str_size += _tpt_header.size;
    str_size += sizeof(u32); // for number of tensors

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        str_size += sizeof(u64); // for str size
        str_size += node->name.size; // for str

        tensor_shape shape = node->tensor->shape;

        str_size += sizeof(u32) * 3; // for width, height, and depth
        str_size += (u64)shape.width * shape.height * shape.depth * sizeof(f32); // for data
    }

    str_buf = MGA_PUSH_ARRAY(arena, u8, str_size);
    if (str_buf == NULL) {
        fprintf(stderr, "Cannot create tensor string: failed to create buffer on arena\n");

        return (string8){ 0 };
    }
    u8* str_buf_ptr = str_buf;

    _WRITE_DATA(_tpt_header.size, _tpt_header.str);
    _WRITE_DATA(sizeof(u32), &list->size);

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        _WRITE_DATA(sizeof(u64), &node->name.size);
        _WRITE_DATA(node->name.size, node->name.str);

        tensor_shape shape = node->tensor->shape;

        _WRITE_DATA(sizeof(u32), &shape.width);
        _WRITE_DATA(sizeof(u32), &shape.height);
        _WRITE_DATA(sizeof(u32), &shape.depth);

        u64 data_size = (u64)shape.width * shape.height * shape.depth * sizeof(f32);

        _WRITE_DATA(data_size, node->tensor->data);
    }

    if (str_size != (u64)(str_buf_ptr - str_buf)) {
        fprintf(stderr, "Cannnot create tensor string: buffer was not filled\n");

        return (string8){ 0 };
    }

    return (string8){ .str = str_buf, .size = str_size };
}

#define _READ_DATA(data_size, data) do { \
        if (pos + (data_size) > str.size) { \
            memset((data), 0, (data_size)); \
        } else { \
            memcpy((data), &str.str[pos], (data_size)); \
            pos += (data_size); \
        } \
    } while (0)

tensor_list tensor_list_from_str(mg_arena* arena, string8 str) {
    if (!str8_equals(_tpt_header, str8_substr(str, 0, _tpt_header.size))) {
        fprintf(stderr, "Cannot read tensor string: tensor header not found\n");
        
        return (tensor_list){ 0 };
    }

    u64 pos = _tpt_header.size;

    tensor_list out = { 0 };
    
    u32 size = 0;
    _READ_DATA(sizeof(u32), &size);

    for (u32 i = 0; i < size; i++) {
        u64 name_size = 0;
        _READ_DATA(sizeof(u64), &name_size);

        string8 name = {
            .size = name_size,
            .str = MGA_PUSH_ZERO_ARRAY(arena, u8, name_size)
        };

        _READ_DATA(name_size, name.str);

        u32 width = 0;
        u32 height = 0;
        u32 depth = 0;

        _READ_DATA(sizeof(u32), &width);
        _READ_DATA(sizeof(u32), &height);
        _READ_DATA(sizeof(u32), &depth);

        tensor* tensor = tensor_create(arena, (tensor_shape){ width, height, depth });
        u64 data_size = (u64)width * height * depth * sizeof(f32);
        _READ_DATA(data_size, tensor->data);

        tensor_list_push(arena, &out, tensor, name);
    }

    if (pos > str.size) {
        fprintf(stderr, "Could not load all tensors: cannot read outisde string bounds\n");
    }

    return out;
}

void tensor_list_save(const tensor_list* list, string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 file_str = tensor_list_to_str(scratch.arena, list);

    if (file_str.size == 0) {
        fprintf(stderr, "Cannnot write tensor file: string was not created\n");
    } else {
        string8_list output_list = { 0 };
        str8_list_push(scratch.arena, &output_list, file_str);

        os_file_write(file_name, output_list);
    }

    mga_scratch_release(scratch);
}
   
tensor_list tensor_list_load(mg_arena* arena, string8 file_name) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 file = os_file_read(scratch.arena, file_name);
    if (file.size == 0) {
        fprintf(stderr, "Cannot load tensors: failed to read file\n");
        
        mga_scratch_release(scratch);
        return (tensor_list){ 0 };
    }

    tensor_list out = tensor_list_from_str(arena, file);

    mga_scratch_release(scratch);

    return out;
}
