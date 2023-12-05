#ifndef TENSOR_H
#define TENSOR_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"

typedef struct {
    ts_u32 width;
    ts_u32 height;
    ts_u32 depth;
} ts_tensor_shape;

typedef struct {
    ts_u32 x, y, z;
} ts_tensor_index;

typedef struct {
    // Size of each dim, 0 if unused
    ts_tensor_shape shape;
    // Number of f32's allocated
    ts_u64 alloc;
    // Data of tensor
    ts_f32* data;
} ts_tensor;

typedef struct ts_tensor_node {
    struct ts_tensor_node* next;

    ts_tensor* tensor;
    ts_string8 name;
} ts_tensor_node;

typedef struct {
    ts_tensor_node* first;
    ts_tensor_node* last;

    ts_u32 size;
} ts_tensor_list;

/*
NOTE:
    All tensor functions expect a height and depth of at least 1
    ONLY CREATE TENSOR WITH ONE OF THE CREATION FUNCTIONS
*/

#ifndef TS_TENSOR_PRINT_IP_ALLOC_ERRORS
#define TS_TENSOR_PRINT_IP_ALLOC_ERRORS 1
#endif

ts_b32 ts_tensor_index_eq(ts_tensor_index a, ts_tensor_index b);
ts_b32 ts_tensor_shape_eq(ts_tensor_shape a, ts_tensor_shape b);

ts_tensor* ts_tensor_create(mg_arena* arena, ts_tensor_shape shape);
ts_tensor* ts_tensor_create_alloc(mg_arena* arena, ts_tensor_shape shape, ts_u64 alloc);
ts_tensor* ts_tensor_copy(mg_arena* arena, const ts_tensor* tensor, ts_b32 keep_alloc);
ts_b32 ts_tensor_copy_ip(ts_tensor* out, const ts_tensor* t);

void ts_tensor_fill(ts_tensor* tensor, ts_f32 num);

ts_tensor_index ts_tensor_argmax(const ts_tensor* t);

ts_b32 ts_tensor_is_zero(const ts_tensor* t);

// Indices work like substring (inclusive start, exclusive end)
ts_tensor* ts_tensor_slice(mg_arena* arena, const ts_tensor* tensor, ts_tensor_index start, ts_tensor_index end);
ts_tensor* ts_tensor_slice_size(mg_arena* arena, const ts_tensor* tensor, ts_tensor_index start, ts_tensor_shape shape);
// Does not copy data
void ts_tensor_2d_view(ts_tensor* out, const ts_tensor* tensor, ts_u32 z);

// Only works for 2d or less
ts_b32 ts_tensor_dot_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
ts_tensor* ts_tensor_dot(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);

// Helper function for computing the output shape of a convolution
ts_tensor_shape ts_tensor_conv_shape(ts_tensor_shape in_shape, ts_tensor_shape kernel_shape, ts_u32 stride_x, ts_u32 stride_y);

// Performs a convolution
// Only works for 2d tensors
ts_b32 ts_tensor_conv_ip(ts_tensor* out, const ts_tensor* input, const ts_tensor* kernel, ts_u32 stride_x, ts_u32 stride_y);
ts_tensor* ts_tensor_conv(mg_arena* arena, const ts_tensor* input, const ts_tensor* kernel, ts_u32 stride_x, ts_u32 stride_y);

// Only works for 2d or less
void ts_tensor_transpose(ts_tensor* t);

ts_b32 ts_tensor_add_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
ts_b32 ts_tensor_sub_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
ts_b32 ts_tensor_component_mul_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
ts_b32 ts_tensor_component_div_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
ts_b32 ts_tensor_scale_ip(ts_tensor* out, const ts_tensor* t, ts_f32 s);
ts_b32 ts_tensor_sqrt_ip(ts_tensor* out, const ts_tensor* t);

ts_tensor* ts_tensor_add(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
ts_tensor* ts_tensor_sub(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
ts_tensor* ts_tensor_component_mul(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
ts_tensor* ts_tensor_component_div(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
ts_tensor* ts_tensor_scale(mg_arena* arena, const ts_tensor* t, ts_f32 s);
ts_tensor* ts_tensor_sqrt(mg_arena* arena, const ts_tensor* t);

void ts_tensor_list_push_existing(ts_tensor_list* list, ts_tensor* tensor, ts_string8 name, ts_tensor_node* node);
void ts_tensor_list_push(mg_arena* arena, ts_tensor_list* list, ts_tensor* tensor, ts_string8 name);
// returns NULL if name is not found in list
ts_tensor* ts_tensor_list_get(const ts_tensor_list* list, ts_string8 name);

ts_string8 ts_tensor_list_to_str(mg_arena* arena, const ts_tensor_list* list);
ts_tensor_list ts_tensor_list_from_str(mg_arena* arena, ts_string8 str);

ts_string8 ts_tensor_get_tpt_header(void);

// *.tpt file
void ts_tensor_list_save(const ts_tensor_list* list, ts_string8 file_name);
ts_tensor_list ts_tensor_list_load(mg_arena* arena, ts_string8 file_name);

#endif // TENSOR_H
