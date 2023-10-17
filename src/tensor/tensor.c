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
        fprintf(stderr, "Cannot create tensor, alloc is too small");
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
            for (u32 x = 0; x < shape.width; x++) {
                out->data[(u64)x + (u64)y * shape.width] += a_data[(u64)i + (u64)y * a_width] * b_data[(u64)x + (u64)i * b_width];
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

File Format:
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

static const string8 file_header = {
    .size = 10,
    .str = (u8*)"TP_tensors"
};

#define _WRITE_DATA(size, data) do { \
        memcpy(file_buf_ptr, (data), (size)); \
        file_buf_ptr += (size); \
    } while (0)

void tensor_list_save(const tensor_list* list, string8 file_name) {
    u64 file_size = 0;
    u8* file_buf = NULL;

    file_size += file_header.size;
    file_size += sizeof(u32); // for number of tensors

    for (tensor_node* node = list->first; node != NULL; node = node->next) {
        file_size += sizeof(u64); // for str size
        file_size += node->name.size; // for str

        tensor_shape shape = node->tensor->shape;

        file_size += sizeof(u32) * 3; // for width, height, and depth
        file_size += (u64)shape.width * shape.height * shape.depth * sizeof(f32); // for data
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    file_buf = MGA_PUSH_ARRAY(scratch.arena, u8, file_size);
    if (file_buf == NULL) {
        fprintf(stderr, "Cannot write file: failed to create buffer on scratch arena\n");

        mga_scratch_release(scratch);

        return;
    }
    u8* file_buf_ptr = file_buf;

    _WRITE_DATA(file_header.size, file_header.str);
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

    if (file_size != (u64)(file_buf_ptr - file_buf)) {
        fprintf(stderr, "Cannnot write file: buffer was not filled\n");
    } else {
        string8_list output_list = { 0 };
        str8_list_push(scratch.arena, &output_list, (string8){ file_size, file_buf });

        os_file_write(file_name, output_list);
    }

    mga_scratch_release(scratch);
}

#define _READ_DATA(data_size, data) do { \
        if (file_pos + (data_size) > file.size) { \
            memset((data), 0, (data_size)); \
        } else { \
            memcpy((data), &file.str[file_pos], (data_size)); \
            file_pos += (data_size); \
        } \
    } while (0)
    
tensor_list tensor_list_load(mg_arena* arena, string8 file_name) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 file = os_file_read(scratch.arena, file_name);
    if (file.size == 0) {
        fprintf(stderr, "Cannot load tensors: failed to read file\n");
        
        mga_scratch_release(scratch);
        return (tensor_list){ 0 };
    }

    if (!str8_equals(file_header, str8_substr(file, 0, file_header.size))) {
        fprintf(stderr, "Cannot load tensors: incorrect file type\n");
        
        mga_scratch_release(scratch);
        return (tensor_list){ 0 };
    }

    u64 file_pos = file_header.size;

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

    if (file_pos > file.size) {
        fprintf(stderr, "Could not load all tensors: cannot read outisde file bounds\n");
    }

    mga_scratch_release(scratch);

    return out;
}
