#include "tensor.h"

#include <stdio.h>
#include <string.h>

b32 tensor_shape_eq(tensor_shape a, tensor_shape b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

tensorf* tensorf_create(mg_arena* arena, tensor_shape shape) {
    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    u64 alloc = (u64)shape.width * shape.height * shape.depth;
    return tensorf_create_alloc(arena, shape, alloc);
}
tensorf* tensorf_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc) {
    if (shape.width == 0) {
        fprintf(stderr, "Unable to create tensorf of width 0\n");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }
    
    u64 min_alloc = (u64)shape.width * shape.height * shape.depth;
    if (alloc < min_alloc) {
        fprintf(stderr, "Cannot create tensorf, alloc is too small");
    }

    tensorf* out = MGA_PUSH_STRUCT(arena, tensorf);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, alloc);
    
    return out;
}
tensorf* tensorf_copy(mg_arena* arena, const tensorf* tensor, b32 keep_alloc) {
    tensor_shape shape = tensor->shape;
    u64 alloc = keep_alloc ? tensor->alloc : ((u64)shape.width * shape.height * shape.depth);

    tensorf* out = MGA_PUSH_STRUCT(arena, tensorf);

    out->shape = shape;
    out->alloc = alloc;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, out->alloc);

    memcpy(out->data, tensor->data, sizeof(f32) * out->alloc);

    return out;
}

void tensorf_fill(tensorf* tensor, f32 num) {
    tensor_shape shape = tensor->shape;
    u64 size = (u64)shape.width * shape.height * shape.depth;

    for (u64 i = 0; i < size; i++) {
        tensor->data[i] = num;
    }
}

tensorf* tensorf_slice(mg_arena* arena, const tensorf* tensor, tensor_index start, tensor_index end) {
    if (end.x > tensor->shape.width || end.y > tensor->shape.height || end.z > tensor->shape.depth) {
        fprintf(stderr, "Cannot create slice past end of tensorf\n");

        return NULL;
    }
    if (start.x > end.x || start.y > end.y || start.z > end.z) {
        fprintf(stderr, "Start of tensorf slice cannot exceed end\n");

        return NULL;
    }
    
    tensor_shape slice_shape = {
        .width = end.x - start.x,
        .height = end.y - start.y,
        .depth = end.z - start.z,
    };

    if (slice_shape.width > tensor->shape.width || slice_shape.height > tensor->shape.height || slice_shape.depth > tensor->shape.depth) {
        fprintf(stderr, "Cannot create slice greater than original tensorf\n");
        
        return NULL;
    }

    tensorf* slice = tensorf_create(arena, slice_shape);
    
    if (slice->shape.depth == 1) { // Fast path for 2d slice of 3d tensor
        u64 start_i = (u64)start.z * tensor->shape.width * tensor->shape.height;

        if (slice->shape.width == tensor->shape.width  && slice->shape.height == tensor->shape.height) {
            u64 slice_size = (u64)tensor->shape.width * tensor->shape.height;

            memcpy(slice->data, &tensor->data[start_i], sizeof(f32) * slice_size);
        } else {
            for (u64 y = start.y; y < end.y; y++) {
                for (u64 x = start.x; x < end.x; x++) {
                    slice->data[x + y * slice->shape.height] = tensor->data[start_i + x + y * tensor->shape.height];
                }
            }
        }
    } else { // General case
        for (u64 z = start.z; z < end.z; z++) {
            for (u64 y = start.y; y < end.y; y++) {
                for (u64 x = start.x; x < end.x; x++) {
                    u64 slice_i = x + y * slice->shape.width + z * slice->shape.width * slice->shape.height;
                    u64 tensor_i = x + y * tensor->shape.width + z * tensor->shape.width * tensor->shape.height;
                    
                    slice->data[slice_i] = tensor->data[tensor_i];
                }
            }
        }
    }

    return slice;
}
tensorf* tensorf_slice_size(mg_arena* arena, const tensorf* tensor, tensor_index start, tensor_shape shape) {
    tensor_index end = {
        start.x + shape.width,
        start.y + shape.height,
        start.z + shape.depth
    };

    return tensorf_slice(arena, tensor, start, end);
}
void tensorf_2d_view(tensorf* out, const tensorf* tensor, u32 z) {
    out->shape = (tensor_shape) {
        .width = tensor->shape.width,
        .height = tensor->shape.height,
        .depth = 1
    };
    out->alloc = (u64)out->shape.width * out->shape.height;

    u64 start_i = (u64)z * tensor->shape.width * tensor->shape.height;

    out->data = &tensor->data[start_i];
}

b32 tensorf_dot_ip(tensorf* out, const tensorf* a, const tensorf* b) {
    if (a->shape.depth != 1 || b->shape.depth != 1) {
        fprintf(stderr, "Cannot dot tensorf in 3 dimensions\n");

        return false;
    }
    if (a->shape.width != b->shape.height) {
        fprintf(stderr, "Cannot dot tensorf: shapes do not align\n");

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
        fprintf(stderr, "Cannot dot tensorf: not enough space in out\n");
        #endif

        return false;
    }

    u32 a_width = a->shape.width;
    u32 a_height = a->shape.height;
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
        for (u32 i = 0; i < a_height; i++) {
            for (u32 x = 0; x < shape.width; x++) {
                out->data[(u64)x + (u64)y * shape.width] += a_data[(u64)i + (u64)y * a_width] * b_data[(u64)x + (u64)i * b_width];
            }
        }
    }

    mga_scratch_release(scratch);

    return true;
}
tensorf* tensorf_dot(mg_arena* arena, const tensorf* a, const tensorf* b) {
    tensor_shape shape = {
        b->shape.width,
        a->shape.height,
        1
    };

    tensorf* out = tensorf_create(arena, shape);

    tensorf_dot_ip(out, a, b);

    return out;
}

b32 tensorf_add_ip(tensorf* out, const tensorf* a, const tensorf* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot add tensorf: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot add tensorf: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    return true;
}
b32 tensorf_sub_ip(tensorf* out, const tensorf* a, const tensorf* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot subtract tensorf: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot subtract tensorf: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    
    return true;
}
b32 tensorf_component_mul_ip(tensorf* out, const tensorf* a, const tensorf* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot multiply tensorf: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot multiply tensorf: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    return true;
}
b32 tensorf_component_div_ip(tensorf* out, const tensorf* a, const tensorf* b) {
    if (!tensor_shape_eq(a->shape, b->shape)) {
        fprintf(stderr, "Cannot divide tensorf: shapes do not align\n");

        return false;
    }
    
    u64 data_size = (u64)a->shape.width * a->shape.height * a->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot divide tensorf: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = a->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }
    
    return true;
}
b32 tensorf_scale_ip(tensorf* out, const tensorf* t, f32 s) {
    u64 data_size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    if (out->alloc < data_size) {
        #if TENSOR_PRINT_IP_ALLOC_ERRORS
        fprintf(stderr, "Cannot scale tensorf: not enough space in out\n");
        #endif

        return false;
    }

    out->shape = t->shape;
    for (u64 i = 0; i < data_size; i++) {
        out->data[i] = t->data[i] * s;
    }

    return true;
}

tensorf* tensorf_add(mg_arena* arena, const tensorf* a, const tensorf* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensorf* out = tensorf_create(arena, a->shape);

    if (!tensorf_add_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensorf* tensorf_sub(mg_arena* arena, const tensorf* a, const tensorf* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensorf* out = tensorf_create(arena, a->shape);

    if (!tensorf_sub_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensorf* tensorf_component_mul(mg_arena* arena, const tensorf* a, const tensorf* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensorf* out = tensorf_create(arena, a->shape);

    if (!tensorf_component_mul_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensorf* tensorf_component_div(mg_arena* arena, const tensorf* a, const tensorf* b) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensorf* out = tensorf_create(arena, a->shape);

    if (!tensorf_component_div_ip(out, a, b)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
tensorf* tensorf_scale(mg_arena* arena, const tensorf* t, f32 s) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    tensorf* out = tensorf_create(arena, t->shape);

    if (!tensorf_scale_ip(out, t, s)) {
        mga_temp_end(maybe_temp);
        
        out = NULL;
    }

    return out;
}
