#include "tensor.h"

#include <stdio.h>
#include <string.h>

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

