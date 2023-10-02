#ifndef TENSOR_H
#define TENSOR_H

#include "base/base.h"
#include "mg/mg_arena.h"

typedef struct {
    u32 width;
    u32 height;
    u32 depth;
} tensor_shape;

typedef struct {
    // Size of each dim, 0 if unused
    tensor_shape shape;

    u64 alloc;
    f32* data;
} tensorf;

#define TENSORF_INDEX(tensor, x, y, z) \
    ((x) + (y) * tensor->shape.width + (z) * tensor->shape.width * tensor->shape.height)

#define TENSORF_AT(tensor, x, y, z) tensor->data[TENSORF_INDEX(tensor, x, y, z)]

tensorf* tensorf_create(mg_arena* arena, tensor_shape shape);

#endif // TENSOR_H
