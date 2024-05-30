/**
 * @file tensor.h
 * @brief 3D tensors
 */

#ifndef TS_TENSOR_H
#define TS_TENSOR_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"

/**
 * @brief Shape of `ts_tensor`
 */
typedef struct {
    /// Width of tensor
    ts_u32 width;
    /// Height of tensor
    ts_u32 height;
    /// Depth of tensor
    ts_u32 depth;
} ts_tensor_shape;

/**
 * @brief Index into ts_tensor
 *
 * Indexing: `Element[x,y,z] == tensor->data[x + y * width + z * width * height]`
 */
typedef struct {
    ts_u32 x, y, z;
} ts_tensor_index;

/**
 * @brief 3D tensor
 */
typedef struct {
    /**
     * @brief Size of each dim
     * 
     * Each should be at least 1 <br>
     * This is ensured in `ts_tensor_create`
     */
    ts_tensor_shape shape;
    /// Number of ts_f32's allocated
    ts_u64 alloc;
    /// Data of tensor
    ts_f32* data;
} ts_tensor;

/**
 * @brief Node in `ts_tensor_list``
 */
typedef struct ts_tensor_node {
    struct ts_tensor_node* next;

    /// Tensor of node
    ts_tensor* tensor;
    /// Name of node
    ts_string8 name;
} ts_tensor_node;

/**
 * @brief List of named tensors
 *
 * You can iterate through the list or
 * get a tensor by name with `ts_tensor_list_get`
 */
typedef struct {
    ts_tensor_node* first;
    ts_tensor_node* last;

    ts_u32 size;
} ts_tensor_list;

/**
 * @brief Whether or not to error if
 *  there is not enough space in out in _ip functions
 */
#ifndef TS_TENSOR_IP_ALLOC_ERRORS
#define TS_TENSOR_IP_ALLOC_ERRORS 1
#endif

/// Returns true if the indices `a` and `b` are equal
ts_b32 ts_tensor_index_eq(ts_tensor_index a, ts_tensor_index b);
/// Returns true if the shapes `a` and `b` are equal
ts_b32 ts_tensor_shape_eq(ts_tensor_shape a, ts_tensor_shape b);

/**
 * @brief Creates a `ts_tensor` and fills it will zero
 *
 * @param arena Arena to allocate `ts_tensor` and data on
 * @param shape Shape of tensor to create.
 *  shape.width MUST be at least 1.
 *  shape.height and shape.depth can be zero.
 *
 * @return The created tensor, filled with zeros
 */
ts_tensor* ts_tensor_create(mg_arena* arena, ts_tensor_shape shape);
/**
 * @brief Creates a tensor with the specified alloc
 *
 * @param arena Arena to allocate `ts_tensor` and data on
 * @param shape Shape of tensor to create.
 *  shape.width MUST be at least 1.
 *  shape.height and shape.depth can be zero.
 * @param alloc Number of `ts_f32`s to allocate.
 *  Must be at least `(ts_u64)shape.width * shape.height * shape.depth`
 *
 * @return The created tensor, filled with zero, and the correct alloc
 */
ts_tensor* ts_tensor_create_alloc(mg_arena* arena, ts_tensor_shape shape, ts_u64 alloc);
/**
 * @brief Copies a `ts_tensor`
 *
 * @param arena Arena to create copy on
 * @param tensor Tensor to copy
 * @param keep_alloc Maintain the alloc of the tensor being copied.
 *  If false, the out alloc is based on the shape of the tensor being copied
 *
 * @return The copied tensor
 */
ts_tensor* ts_tensor_copy(mg_arena* arena, const ts_tensor* tensor, ts_b32 keep_alloc);
/**
 * @brief Copies `t` into `out` if `out` is big enough
 *
 * @param out Where `t` gets coppied
 * @param t Tensor to copy
 *
 * @return true if `out` is big enough, `false` otherwise
 */
ts_b32 ts_tensor_copy_ip(ts_tensor* out, const ts_tensor* t);

/// Fills `tensor` with `num`
void ts_tensor_fill(ts_tensor* tensor, ts_f32 num);

/// Returns the index of the maximum element of `t`
ts_tensor_index ts_tensor_argmax(const ts_tensor* t);

/// Returns true if `t` is all zero
ts_b32 ts_tensor_is_zero(const ts_tensor* t);

/**
 * @brief Gets a 2D view from a 3D tensor. DOES NOT COPY THE DATA
 *
 * @param out Output of view
 * @param tensor Tensor you are viewing
 * @param z Index of 2D slice
 */
void ts_tensor_2d_view(ts_tensor* out, const ts_tensor* tensor, ts_u32 z);

/**
 * @brief Computes the dot product of `a` and `b`.
 *
 * `a` and `b` have to be 2D.
 * `a.width` must equal `b.height`
 *
 * @param out Output of dot product. Needs to be big enough (i.e. (b.width, a.height, 1))
 * @param transpose_a Whether or not to transpose a
 * @param transpose_b Whether or not to transpose b
 * @param a First tensor
 * @param b Second tensor
 *
 * @return true if `out` was big enough, false otherwise
 */
ts_b32 ts_tensor_dot_ip(ts_tensor* out, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b);
/**
 * @brief Computes the dot product of `a` and `b`. Must be 2D tensors (depth == 1)
 *
 * See `ts_tensor_dot_ip` for more
 */
ts_tensor* ts_tensor_dot(mg_arena* arena, ts_b32 transpose_a, ts_b32 transpose_b, const ts_tensor* a, const ts_tensor* b);

/**
 * @brief Computes the output shape of `ts_tensor_cov`
 *
 * See ts_tensor_conv for more detail
 */
ts_tensor_shape ts_tensor_conv_shape(ts_tensor_shape in_shape, ts_tensor_shape kernel_shape, ts_u32 stride_x, ts_u32 stride_y);

/**
 * @brief Implements the famous `im2col` function. In place version
 *
 * Converts 3d sections of the input image into rows in the output image. <br>
 * `input` and `out` cannot be the same or overlap. <br>
 * Commonly used in convolutional layers to speed up convolutions
 *
 * @param out Output rows
 * @param input Input image
 * @param kernel_size Side length of kernel
 * @param stride Stride of convolution
 * @param padding Padding of image on each side of x and y
 *
 * @return true if `out` is big enough
 */
ts_b32 ts_tensor_im2col_ip(ts_tensor* out, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding);
/**
 * @brief Implements the `im2col` function
 *
 * See `ts_tensor_im2col_ip` for details
 */
ts_tensor* ts_tensor_im2col(mg_arena* arena, const ts_tensor* input, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding);

/**
 * @brief Implements the famous `col2im` function. In place version
 *
 * Converts rows of input matrix into an image. 
 * Used in convolution layers
 *
 * @param out Output image
 * @param input 2D input matrix
 * @param out_shape Shape of output image (width, height, channels)
 * @param kernel_size Side length of kernel
 * @param stride Stride of convolution
 * @param padding Padding of image on each side of x and y
 *
 * @return true if `out` is big enough
 */
ts_b32 ts_tensor_col2im_ip(ts_tensor* out, const ts_tensor* input, ts_tensor_shape out_shape, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding);
/**
 * @brief Implements the famour `col2im` function.
 *
 * See `ts_tensor_col2im` for details
 */
ts_tensor* ts_tensor_col2im(mg_arena* arena, const ts_tensor* input, ts_tensor_shape out_shape, ts_u32 kernel_size, ts_u32 stride, ts_u32 padding);

/**
 * @brief Transposes a 2D tensor in place
 *
 * Must be 2D
 *
 * @return true on success, false otherwise
 */
ts_b32 ts_tensor_transpose_ip(ts_tensor* t);
/**
 * @brief Creates a transposed version of `t`
 */
ts_tensor* ts_tensor_transpose(mg_arena* arena, const ts_tensor* t);

/**
 * @brief Adds `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_add_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
/**
 * @brief Subtracts `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_sub_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
/**
 * @brief Component multiplies `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_component_mul_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
/**
 * @brief Component divides `a` and `b` into out
 *
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_component_div_ip(ts_tensor* out, const ts_tensor* a, const ts_tensor* b);
/**
 * @brief Scales `t` by `s`
 *
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_scale_ip(ts_tensor* out, const ts_tensor* t, ts_f32 s);
/**
 * @brief Computes the square root of `t`
 * 
 * @return true if `out` is big enough, false otherwise
 */
ts_b32 ts_tensor_sqrt_ip(ts_tensor* out, const ts_tensor* t);

/// Creates a `ts_tensor` that is the sum of `a` and `b`
ts_tensor* ts_tensor_add(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
/// Creates a `ts_tensor` that is the difference of `a` and `b`
ts_tensor* ts_tensor_sub(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
/// Creates a `ts_tensor` that is the component product of `a` and `b`
ts_tensor* ts_tensor_component_mul(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
/// Creates a `ts_tensor` that is the component quotient of `a` and `b`
ts_tensor* ts_tensor_component_div(mg_arena* arena, const ts_tensor* a, const ts_tensor* b);
/// Creates a `ts_tensor` that is `t` scaled by `s`
ts_tensor* ts_tensor_scale(mg_arena* arena, const ts_tensor* t, ts_f32 s);
/// Creates a `ts_tensor` that is the square root of `t`
ts_tensor* ts_tensor_sqrt(mg_arena* arena, const ts_tensor* t);

/**
 * @brief Pushes a `ts_tensor` and `ts_string8` name to a `ts_tensor_list`
 *  with an existing `ts_tensor_node`
 *
 * @param list List to push to
 * @param tensor Tensor to push onto
 * @param name Name of tensor being pushed
 * @param node Node to push
 */
void ts_tensor_list_push_existing(ts_tensor_list* list, ts_tensor* tensor, ts_string8 name, ts_tensor_node* node);
/**
 * @brief Pushes a `ts_tensor` and `ts_string8` name to a `ts_tensor_list`
 *
 * Does not copy `tensor`
 *
 * @param arena Arena to create node on
 * @param list List to push onto
 * @param tensor Tensor to push
 * @param name Name of tensor being pushed
 */
void ts_tensor_list_push(mg_arena* arena, ts_tensor_list* list, ts_tensor* tensor, ts_string8 name);
/**
 * @brief Gets a `ts_tensor` from a `ts_tensor_list` with a name
 *
 * @param list List to get from
 * @param name Name of tensor to get
 *
 * @return `ts_tensor` corresponding to `name`, or NULL if `name` is not in list
 */
ts_tensor* ts_tensor_list_get(const ts_tensor_list* list, ts_string8 name);

/**
 * @brief Serializes a `ts_tensor_list` to a `ts_string8`
 *
 * Serializes according to the .tst format. 
 * See `ts_tensor_list_save` in `tensor.c` for more
 *
 * @param arena Arena to create `ts_string8` on
 * @param list List to serialize
 *
 * @return Serialized list
 */
ts_string8 ts_tensor_list_to_str(mg_arena* arena, const ts_tensor_list* list);
/**
 * @brief Creates a `ts_tensor_list` from a `ts_string8`
 *
 * @param arena Arena to push `ts_tensor`s and `ts_tensor_node`s onto
 * @param str String to load
 *
 * @return List of tensors from the string
 */
ts_tensor_list ts_tensor_list_from_str(mg_arena* arena, ts_string8 str);

/// Returns the .tst file header 
ts_string8 ts_tensor_get_tst_header(void);

/**
 * @brief Serializes a `ts_tensor_list` into a file according to the .tst file format
 *
 * See `tensor.c` for more about the format
 *
 * @param list List to save
 * @param file_name Output file. Include file extention in `file_name`
 */
void ts_tensor_list_save(const ts_tensor_list* list, ts_string8 file_name);
/**
 * @brief Loads a `ts_tensor_list` from a file
 *
 * @param arena Arena to push `ts_tensor`s and `ts_tensor_node`s onto
 * @param file_name File to load
 *
 * @return List of tensors from file
 */
ts_tensor_list ts_tensor_list_load(mg_arena* arena, ts_string8 file_name);

#endif // TS_TENSOR_H

