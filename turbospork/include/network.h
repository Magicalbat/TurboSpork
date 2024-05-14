/**
 * @file network.h
 * @brief Neural networks
 */

#ifndef TS_NETWORK_H
#define TS_NETWORK_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"
#include "tensor.h"

#include "layers.h"
#include "costs.h"
#include "optimizers.h"

/**
 * @brief Sequential neural network
 */
typedef struct {
    /// Whether or not training mode is enabled. Set during creation functions
    ts_b32 training_mode;

    /// Number of layers
    ts_u32 num_layers;
    /// Array of layers
    ts_layer** layers;

    /**
     * @brief List of layer descs
     * 
     * Used for neural network saving
     */
    ts_layer_desc* layer_descs;

    /**
     * @brief Used for forward and backward passes
     * Allows for single allocation of input/output variable
     */
    ts_u64 max_layer_size;
} ts_network;

/// Information about random transformations in the network training inputs
typedef struct {
    /// Minimum random translation, inclusize. Applied on both axes
    ts_f32 min_translation;
    /// Maximum random translation, exclusize. Applied on both axes
    ts_f32 max_translation;

    /// Minimum random scale, inclusive. Applied on both axes
    ts_f32 min_scale;
    /// Maximum random scale, exclusive. Applied on both axes
    ts_f32 max_scale;

    /// Minimum random angle in radians, inclusive
    ts_f32 min_angle;
    /// Maximum random angle in radians, exclusive
    ts_f32 max_angle;
} ts_network_transforms;

/// Info for epoch callback
typedef struct {
    /// Epoch number. Starts at 0
    ts_u32 epoch;

    /// Accuracy of test, if accuracy test is enabled in training
    ts_f32 test_accuracy;
} ts_network_epoch_info;

/// Callback function for training
typedef void(ts_network_epoch_callback)(const ts_network_epoch_info*);

/**
 * @brief Neural network training description
 *
 * For `ts_network_train`
 */
typedef struct {
    /// Number of epochs to train
    ts_u32 epochs;
    /// Size of training batch
    ts_u32 batch_size;

    /// Number of threads to train on
    ts_u32 num_threads;

    /// Cost function to use
    ts_cost_type cost;
    /**
     * @brief Optimizer to use
     *
     * You do not have to set `batch_size` in the optimizer
     */
    ts_optimizer optim;

    /// Whether or not to randomly transform the training inputs
    ts_b32 random_transforms;
    /// Random transforms to be applied to training inputs
    ts_network_transforms transforms;


    /// Callback function called after each epoch. Can be NULL 
    ts_network_epoch_callback* epoch_callback;

    /**
     * @brief Epoch interval to save network
     *
     * If `save_interval` == 0, then the network does not save. <br>
     * Saves when `(epoch + 1) % save_interval == 0`
     */
    ts_u32 save_interval;
    /**
     * @brief Output path of save interval
     *
     * Output file is `{save_path}{epoch}.tsn`
     */
    ts_string8 save_path;

    /**
     * @brief Training inputs to neural network.
     *
     * One training input is a 2D slice of the `train_inputs` tensor.
     * If you have 3D inputs, reduce them to 2D for `train_inputs`
     * then resize them in the input layer of the neural network
     */
    ts_tensor* train_inputs;
    /**
     * @brief Training outputs of neural network.
     *
     * 2D slices are taken for each output.
     * Depth must be the same as the depth of `train_inputs`
     */
    ts_tensor* train_outputs;

    /// Whether or not to enable an accuracy test after each epoch
    ts_b32 accuracy_test;
    /**
     * @brief Inputs for testing
     *
     * Same shape requirements as `train_inputs` apply
     */
    ts_tensor* test_inputs;
    /**
     * @brief Outputs for testing
     *
     * Same shape requirements as `train_outputs` apply
     */
    ts_tensor* test_outputs;
} ts_network_train_desc;

// This training_mode overrides the one in the desc
/**
 * @brief Creates a neural network
 *
 * @param arena Arena to create network on
 * @param num_layers Number of layers and size of the `layer_descs` array
 * @param layer_descs List of layer descriptions
 * @param training_mode Whether or not to initialize the network in training mode.
 *  This overrides the training mode in the layer descs
 *
 * @return Pointer to network on success, NULL on failure
 */
ts_network* ts_network_create(mg_arena* arena, ts_u32 num_layers, const ts_layer_desc* layer_descs, ts_b32 training_mode);
/**
 * @brief Creates a network from a layout file (.tsl)
 *
 * Layout files can be created by hand or by `ts_network_save_layout`
 *
 * @param arena Arena to create network on
 * @param file_name File to load
 * @param training_mode Whether or not to initalize the network in training mode
 *
 * @return Pointer to network on success, NULL on failure
 */
ts_network* ts_network_load_layout(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode);
/**
 * @brief Creates a network from a network file (.tsn)
 *
 * Network files are created by `ts_network_save`,
 * and they include the parameters of the neural network.
 * Used to load a network that has already been trained.
 *
 * @param arena Arena to create network on
 * @param file_name File to load
 * @param training_mode Whether or not to initalize the network in training mode
 *
 * @return Pointer to network on success, NULL on failure
 */
ts_network* ts_network_load(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode);

void ts_network_load_existing(ts_network* nn, ts_string8 file_name);

/**
 * @brief Deletes the neural network
 *
 * This is annoying, but required because of some threading stuff
 */
void ts_network_delete(ts_network* nn);

/**
 * @brief Feeds `input` through the network, and puts the result in `out`
 *
 * @param nn Network to use
 * @param out Output of feedforward. Must be big enough
 * @param input Input to network
 */
void ts_network_feedforward(const ts_network* nn, ts_tensor* out, const ts_tensor* input);

/**
 * @brief Trains the neural network based on the training description
 *
 * See `ts_network_train_desc` for details
 *
 * @param nn Network to train
 * @param desc Training description
 */
void ts_network_train(ts_network* nn, const ts_network_train_desc* desc);

/**
 * @brief Prints a summary of the network to stdout
 *
 * Shows the layer types and shapes
 */
void ts_network_summary(const ts_network* nn);

/**
 * @brief Saves the layout of the network into a .tsl file
 *
 * Saves any information stored in the layer descriptions
 *
 * @param nn Network to save layout
 * @param file_name Output of save layout. This should include the file extension
 */
void ts_network_save_layout(const ts_network* nn, ts_string8 file_name);

/**
 * @brief Saves the network into a .tsn file
 *
 * Saves layout and parameter information.
 * Usually used during or after training the network
 *
 * @param nn Network to save
 * @param file_name File to save to. This shoudl include the file extension
 */
void ts_network_save(const ts_network* nn, ts_string8 file_name);

#endif // TS_NETWORK_H

