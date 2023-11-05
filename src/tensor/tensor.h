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
} tensor;

typedef struct tensor_node {
    struct tensor_node* next;

    tensor* tensor;
    string8 name;
} tensor_node;

typedef struct {
    tensor_node* first;
    tensor_node* last;

    u32 size;
} tensor_list;

/*
NOTE:
    All tensor functions expect a height and depth of at least 1
    ONLY CREATE TENSOR WITH ONE OF THE CREATION FUNCTIONS
*/

#ifndef TENSOR_PRINT_IP_ALLOC_ERRORS
#define TENSOR_PRINT_IP_ALLOC_ERRORS 1
#endif

#define tensor_INDEX(tensor, x, y, z) \
    ((u64)(x) + (u64)(y) * tensor->shape.width + (u64)(z) * tensor->shape.width * tensor->shape.height)
#define tensor_AT(tensor, x, y, z) tensor->data[tensor_INDEX(tensor, x, y, z)]

b32 tensor_shape_eq(tensor_shape a, tensor_shape b);

tensor* tensor_create(mg_arena* arena, tensor_shape shape);
tensor* tensor_create_alloc(mg_arena* arena, tensor_shape shape, u64 alloc);
tensor* tensor_copy(mg_arena* arena, const tensor* tensor, b32 keep_alloc);
b32 tensor_copy_ip(tensor* out, const tensor* t);

void tensor_fill(tensor* tensor, f32 num);

tensor_index tensor_argmax(const tensor* t);

b32 tensor_is_zero(const tensor* t);

// Indices work like substring (inclusive start, exclusive end)
tensor* tensor_slice(mg_arena* arena, const tensor* tensor, tensor_index start, tensor_index end);
tensor* tensor_slice_size(mg_arena* arena, const tensor* tensor, tensor_index start, tensor_shape shape);
// Does not copy data
void tensor_2d_view(tensor* out, const tensor* tensor, u32 z);

// Only works for 2d or less
b32 tensor_dot_ip(tensor* out, const tensor* a, const tensor* b);
tensor* tensor_dot(mg_arena* arena, const tensor* a, const tensor* b);

// Only works for 2d or less
void tensor_transpose(tensor* t);

b32 tensor_add_ip(tensor* out, const tensor* a, const tensor* b);
b32 tensor_sub_ip(tensor* out, const tensor* a, const tensor* b);
b32 tensor_component_mul_ip(tensor* out, const tensor* a, const tensor* b);
b32 tensor_component_div_ip(tensor* out, const tensor* a, const tensor* b);
b32 tensor_scale_ip(tensor* out, const tensor* t, f32 s);
b32 tensor_sqrt_ip(tensor* out, const tensor* t);

tensor* tensor_add(mg_arena* arena, const tensor* a, const tensor* b);
tensor* tensor_sub(mg_arena* arena, const tensor* a, const tensor* b);
tensor* tensor_component_mul(mg_arena* arena, const tensor* a, const tensor* b);
tensor* tensor_component_div(mg_arena* arena, const tensor* a, const tensor* b);
tensor* tensor_scale(mg_arena* arena, const tensor* t, f32 s);
tensor* tensor_sqrt(mg_arena* arena, const tensor* t);

void tensor_list_push_existing(tensor_list* list, tensor* tensor, string8 name, tensor_node* node);
void tensor_list_push(mg_arena* arena, tensor_list* list, tensor* tensor, string8 name);
// returns NULL if name is not found in list
tensor* tensor_list_get(const tensor_list* list, string8 name);

string8 tensor_list_to_str(mg_arena* arena, const tensor_list* list);
tensor_list tensor_list_from_str(mg_arena* arena, string8 str);

string8 tensor_get_tpt_header(void);

// *.tpt file
void tensor_list_save(const tensor_list* list, string8 file_name);
tensor_list tensor_list_load(mg_arena* arena, string8 file_name);

#endif // TENSOR_H
