/**
 * @file layers.h
 * @brief Neural network layers
 */

#ifndef LAYERS_H
#define LAYERS_H

#include "mg/mg_arena.h"

#include "base_defs.h"
#include "str.h"
#include "tensor.h"
#include "optimizers.h"

/**
 * @brief Type of layer
 */
typedef enum {
    /// Does not do anything, should not be used
    TS_LAYER_NULL = 0,
    /// Must be first layer of neural network
    TS_LAYER_INPUT,
    /// Reshapes input of layer and delta of backprop
    TS_LAYER_RESHAPE,
    /// Fully connected layer
    TS_LAYER_DENSE,
    /// Applies activation function to input and multiplies activation gradient with delta
    TS_LAYER_ACTIVATION,
    /// Randomly turns off some neurons during training
    TS_LAYER_DROPOUT,
    /// Reshapes input to 1D
    TS_LAYER_FLATTEN,
    /// Pools all 2D slices of a 3D input
    TS_LAYER_POOLING_2D,
    /// 2D convolution layer
    TS_LAYER_CONV_2D,

    /// Number of layers
    TS_LAYER_COUNT
} ts_layer_type;

/**
 * @brief Parameter initialization types
 *
 * Layers will initialize trainable parameters with one of these types <br>
 * See `ts_param_init` for more 
 */
typedef enum {
    /// Does nothing to param
    TS_PARAM_INIT_NULL = 0,

    /// Fills param with zeors
    TS_PARAM_INIT_ZEROS,
    /// Fills param with zeors
    TS_PARAM_INIT_ONES,

    /// Xavier Glorot uniform
    TS_PARAM_INIT_XAVIER_UNIFORM,
    /// Xavier Glorot normal
    TS_PARAM_INIT_XAVIER_NORMAL,

    /// He/Kaiming uniform
    TS_PARAM_INIT_HE_UNIFORM,
    /// He/Kaiming normal
    TS_PARAM_INIT_HE_NORMAL,

    /// Number of param init types
    TS_PARAM_INIT_COUNT
} ts_param_init_type;

/**
 * @brief Activation types for activation layer
 */
typedef enum {
    /// Does nothing to inputs
    TS_ACTIVATION_NULL = 0,
    /// Linear function
    TS_ACTIVATION_LINEAR,
    /// Sigmoid
    TS_ACTIVATION_SIGMOID,
    /// Tanh
    TS_ACTIVATION_TANH,
    /// Relu
    TS_ACTIVATION_RELU,
    /// Leaky relu with leaky value of 0.01
    TS_ACTIVATION_LEAKY_RELU,
    /// Softmax
    TS_ACTIVATION_SOFTMAX,

    /// Number of activation functions
    TS_ACTIVATION_COUNT
} ts_layer_activation_type;

/**
 * @brief Pooling types for pooling layer
 */
typedef enum {
    /// Does nothing
    TS_POOLING_NULL = 0,
    /// Max pooling
    TS_POOLING_MAX,
    /// Average pooling
    TS_POOLING_AVG,

    /// NUmber of pooling types
    TS_POOLING_COUNT
} ts_layer_pooling_type;


/// Input layer description 
typedef struct {
    /**
     * @brief Shape of input layer
     *
     * Will reshape the neural network input to this shape
     */
    ts_tensor_shape shape;
} ts_layer_input_desc;

/// Reshape layer description
typedef struct {
    /**
     * @brief Shape of layer output
     *
     * Will reshape layer input to shape and backprop delta to input shape
     */
    ts_tensor_shape shape;
} ts_layer_reshape_desc;

/**
 * @brief Dense layer description
 *
 * Output shape is (`size`, 1, 1)
 */
typedef struct {
    /// Output size of layer
    ts_u32 size;

    /**
     * @brief Initialization type for bias
     *
     * Defaults to TS_PARAM_INIT_ZEROS
     */
    ts_param_init_type bias_init;

    /**
     * @brief Initialization type for weight
     *
     * Defaults to TS_PARAM_INIT_XAVIER_UNIFORM
     */
    ts_param_init_type weight_init;
} ts_layer_dense_desc;

/**
 * @brief Activation layer description
 *
 * Activation layers maintain the previous layer's shape
 */
typedef struct {
    /**
     * @brief Which activation function to use
     *
     * Defaults to TS_ACTIVATION_RELU
     */
    ts_layer_activation_type type;
} ts_layer_activation_desc;

/**
 * @brief Dropout layer description
 *
 * Random dropout is only applied during training <br>
 * Dropout layers maintain the previous layer's shape
 */
typedef struct {
    /// Keeprate for dropout
    ts_f32 keep_rate;
} ts_layer_dropout_desc;

/**
 * @brief 2D Pooling layer description
 */
typedef struct {
    /**
     * @brief Size of pooling
     *
     * depth of `pool_size` is ignored
     */
    ts_tensor_shape pool_size;

    /**
     * @brief Type of pooling to use
     *
     * Defaults to TS_POOLING_MAX
     */
    ts_layer_pooling_type type;
} ts_layer_pooling_2d_desc;

/**
 * @brief 2D Convolutional layer description
 */
typedef struct {
    /**
     * @brief Number of output filters
     *
     * Depth of output shape will equal `num_filters`
     */
    ts_u32 num_filters;

    /**
     * @brief Side length of kernel for convolution operation
     */
    ts_u32 kernel_size;

    /**
     * @brief Adds padding to input before the convolution operation
     *
     * The output size will equal the input size if
     * the strides are 1 and padding is true
     */
    ts_b32 padding;

    /// Stride for convolution. Defaults to 1
    ts_u32 stride;

    /**
     * @brief Initialization type for kernels
     *
     * Defaults to TS_PARAM_INIT_HE_NORMAL
     */
    ts_param_init_type kernels_init;

    /**
     * @brief Initialization type for biases
     *
     * Defaults to TS_PARAM_INIT_ZEROS
     */

    ts_param_init_type biases_init;
} ts_layer_conv_2d_desc;

/**
 * @brief Full layer description
 */ 
typedef struct {
    /**
     * @brief Type of layer
     *
     * Used to determine which member of the union is used
     */
    ts_layer_type type;
    /**
     * @brief Used to determine if layer should be created for training
     *
     * Training mode uses more memory, but is necessary for training the network. <br>
     * Only use it when training the network
     */
    ts_b32 training_mode;

    union {
        /// Input desc
        ts_layer_input_desc input;
        /// Reshape desc
        ts_layer_reshape_desc reshape;
        /// Dense desc
        ts_layer_dense_desc dense;
        /// Activation desc
        ts_layer_activation_desc activation;
        /// Dropout desc
        ts_layer_dropout_desc dropout;
        /// Pooling2D desc
        ts_layer_pooling_2d_desc pooling_2d;
        /// Convolutional2D desc
        ts_layer_conv_2d_desc conv_2d;
    };
} ts_layer_desc;

/**
 * @brief Layer structure
 *
 * Defined in layers_internal.h (in src)
 */
typedef struct ts_layer ts_layer;

/// Node for `ts_layers_cache` singly linked list
typedef struct ts_layers_cache_node {
    ts_tensor* t;
    struct ts_layers_cache_node* next;
} ts_layers_cache_node;

/**
 * @brief Layers cache
 *
 * This is just a stack of `ts_tensor`s used in layer feedforward and backprop functions. <br>
 * Layers use the cache if they need to transfer data from the feedforward to the backprop. <br>
 * This is necessary because of the multithreading.
 */
typedef struct {
    /**
     * @brief Arena used for the cache
     * 
     * If a layer is pushing tensors onto the cache
     * the tensor should created with this arena
     */
    mg_arena* arena;

    /// First node of SLL
    ts_layers_cache_node* first;
    /// Last node of SLL
    ts_layers_cache_node* last;
} ts_layers_cache;

/**
 * @brief Gets the name of a layer from the type
 * 
 * @return `ts_string8` with the layer name. Do not modify the string data 
 */
ts_string8 ts_layer_get_name(ts_layer_type type);
/**
 * @brief Gets hte layer type from the `name`
 *
 * @return `ts_layer_type` correlated with `name`; TS_LAYER_NULL if `name` is null or invalid
 */
ts_layer_type ts_layer_from_name(ts_string8 name);

/**
 * @brief Creates a layer from a `ts_layer_desc`
 *
 * @param arena Memory arena to create the layer in
 * @param desc Pointer to layer desc
 * @param prev_shape Shape of previous layer. This is required for many layers to work properly
 */
ts_layer* ts_layer_create(mg_arena* arena, const ts_layer_desc* desc, ts_tensor_shape prev_shape);
/**
 * @brief Feedforwards layer
 *
 * @param l Layer to be used
 * @param in_out Input to layer and where the output gets stored
 * @param cache Layer cache only used for training. Can be NULL
 */
void ts_layer_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache); 
/**
 * @brief Backpropagation of layer
 *
 * Layer should be in training mode, and the cache is required
 *
 * @param l Layer to be used
 * @param delta Running gradient of backpropagation.
 *  The backprop function will update any layer params and the delta
 * @param cache Layer cache
 */
void ts_layer_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache);
/**
 * @brief Applies any changes accumulated in backprop to layer
 *
 * @param l Layer to be used
 * @param optim Optimizer to be uzed
 */
void ts_layer_apply_changes(ts_layer* l, const ts_optimizer* optim);
/**
 * @brief Deletes the layer
 *
 * This is annoying, but it is required for some multithreading stuff.
 *
 * @param l Layer to delete
 */
void ts_layer_delete(ts_layer* l);
// Saves layer params, not anything that would be in the desc
/** 
 * @brief Saves any trainable params of the layer
 *
 * This does not include anything that would be in a desc.
 *
 * @param arena Arena for nodes in the `list`
 * @param l Layer to save
 * @param list List to save tensors to
 * @param index Index of the layer in neural network. To make names in the list unique
 */
void ts_layer_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index);
// Loads layer params, not anything that would be in the desc
// Layer needs to be created with the correct type
/**
 * @brief Loads trainable params of the layer
 *
 * @param l The layer to load. The layer should be initialized with a desc
 * @param list List with the loaded tensors
 * @param index Index of the layer in the neural network
 */
void ts_layer_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index);

/**
 * @brief Retrives the default desc of the layer type
 *
 * @return A copy of the default layer desc. It is okay to modify the return value.
 */
ts_layer_desc ts_layer_desc_default(ts_layer_type type);
/**
 * @brief Applies defaults to parameters in the desc
 * 
 * @param desc A layer desc with the type set
 *
 * @return A new layer desc with default values for unset members of `desc`
 */
ts_layer_desc ts_layer_desc_apply_default(const ts_layer_desc* desc);

/**
 * @brief Saves the desc to the `ts_string8_list`
 *
 * Example format: `layer_type: field = value;`
 *
 * @param arena Arena for strings and nodes on the list
 * @param list Output string list for saving
 * @param desc Desc to save
 */
void ts_layer_desc_save(mg_arena* arena, ts_string8_list* list, const ts_layer_desc* desc);
/**
 * @brief Loads the layer desc from the `ts_string8`
 * 
 * @param str A valid layer desc str
 */
ts_layer_desc ts_layer_desc_load(ts_string8 str);

/**
 * @brief Initializes `param` based on the init type
 *
 * @param param The tensor to init
 * @param input_type Type of initialization
 * @param in_size Size of input to param/layer (e.g. `(ts_u64)input->shape.width * input->shape.height * input->shape.depth`)
 * @param out_size Size of output of param/layer
 */
void ts_param_init(ts_tensor* param, ts_param_init_type input_type, ts_u64 in_size, ts_u64 out_size);

/**
 * @brief Pushes the `ts_tensor` onto the `ts_layers_cache`
 */
void ts_layers_cache_push(ts_layers_cache* cache, ts_tensor* t);
/** 
 * @brief Pops a `ts_tensor` off of the `ts_layers_cache` and returns it
 */
ts_tensor* ts_layers_cache_pop(ts_layers_cache* cache);

#endif // LAYERS_H

