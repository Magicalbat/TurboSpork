#include "tensor.h"

#include <stdio.h>

tensorf* tensorf_create(mg_arena* arena, tensor_shape shape) {
    if (shape.width == 0) {
        fprintf(stderr, "Unable to create tensor of width 0\n");
        return NULL;
    }

    if (shape.height == 0) { shape.height = 1; }
    if (shape.depth == 0) { shape.depth = 1; }

    tensorf* out = MGA_PUSH_STRUCT(arena, tensorf);

    out->shape = shape;
    out->alloc = (u64)shape.width * shape.height * shape.depth;
    out->data = MGA_PUSH_ZERO_ARRAY(arena, f32, out->alloc);
    
    return out;
}

