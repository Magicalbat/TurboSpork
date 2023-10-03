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
    u32 x, y, z;
} tensor_index;

typedef struct {
    // Size of each dim, 0 if unused
    tensor_shape shape;
    // Number of f32's allocated
    u64 alloc;
    // Data of tensor
    f32* data;
} tensorf;

/*
NOTE:
    All tensorf functions expect a height and depth of at least 1
    ONLY CREATE TENSOR WITH ONE OF THE CREATION FUNCTIONS
*/

#define TENSORF_INDEX(tensor, x, y, z) \
    ((u64)(x) + (u64)(y) * tensor->shape.width + (u64)(z) * tensor->shape.width * tensor->shape.height)
#define TENSORF_AT(tensor, x, y, z) tensor->data[TENSORF_INDEX(tensor, x, y, z)]

b32 tensor_shape_eq(tensor_shape a, tensor_shape b);

tensorf* tensorf_create(mg_arena* arena, tensor_shape shape);
tensorf* tensorf_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc);
tensorf* tensorf_copy(mg_arena* arena, const tensorf* tensor, b32 keep_alloc);

void tensorf_fill(tensorf* tensor, f32 num);

// Indices work like substring (inclusive start, exclusive end)
tensorf* tensorf_slice(mg_arena* arena, const tensorf* tensor, tensor_index start, tensor_index end);
tensorf* tensorf_slice_size(mg_arena* arena, const tensorf* tensor, tensor_index start, tensor_shape shape);
// Does not copy data
void tensorf_2d_view(tensorf* out, const tensorf* tensor, u32 z);

#ifndef TENSOR_PRINT_IP_ALLOC_ERRORS
#define TENSOR_PRINT_IP_ALLOC_ERRORS 1
#endif

b32 tensorf_add_ip(tensorf* out, const tensorf* a, const tensorf* b);
b32 tensorf_sub_ip(tensorf* out, const tensorf* a, const tensorf* b);
b32 tensorf_component_mul_ip(tensorf* out, const tensorf* a, const tensorf* b);
b32 tensorf_component_div_ip(tensorf* out, const tensorf* a, const tensorf* b);
b32 tensorf_scale_ip(tensorf* out, const tensorf* t, f32 s);

tensorf* tensorf_add(mg_arena* arena, const tensorf* a, const tensorf* b);
tensorf* tensorf_sub(mg_arena* arena, const tensorf* a, const tensorf* b);
tensorf* tensorf_component_mul(mg_arena* arena, const tensorf* a, const tensorf* b);
tensorf* tensorf_component_div(mg_arena* arena, const tensorf* a, const tensorf* b);
tensorf* tensorf_scale(mg_arena* arena, const tensorf* t, f32 s);

#endif // TENSOR_H
