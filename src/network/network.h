#ifndef NETWORK_H
#define NETWORK_H

#include "base/base.h"
#include "mg/mg_arena.h"
#include "tensor/tensor.h"

#include "layers/layers.h"
#include "costs/costs.h"
#include "optimizers/optimizers.h"

typedef struct {
    u32 num_layers;
    layer** layers;

    // For saving
    layer_desc* layer_descs;

    // Used for forward and backward passes
    // Allows for single allocation of input/output variable
    u64 max_layer_size;
} network;

typedef struct {
    u32 epoch;

    f32 test_accuracy;
} network_epoch_info;

typedef void(network_epoch_callback)(const network_epoch_info*);

typedef struct {
    u32 epochs;
    u32 batch_size;

    u32 num_threads;

    cost_type cost;
    optimizer optim;

    // Can be null
    // Gives information to function after each epoch
    network_epoch_callback* epoch_callback;

    // Epoch interval to save network 
    // When (epoch + 1) % save_interval == 0
    // Interval of zero means no saving
    u32 save_interval;
    // Output will be "{save_path}{epoch_num}.tpn"
    string8 save_path;

    tensor* train_inputs;
    tensor* train_outputs;

    b32 accuracy_test;
    tensor* test_inputs;
    tensor* test_outputs;
} network_train_desc;

// This training_mode overrides the one in the desc
network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode);
// Reads layout file (*.tpl)
// See network_layout_save
network* network_load_layout(mg_arena* arena, string8 file_name, b32 training_mode);
// Reads network file (*.tpn)
// See network_save
network* network_load(mg_arena* arena, string8 file_name, b32 training_mode);

void network_delete(network* nn);

void network_feedforward(const network* nn, tensor* out, const tensor* input);
void network_train(network* nn, const network_train_desc* desc);

// Prints the network summary to stdout
void network_summary(const network* nn);

// Saves layer descs
// *.tpl
void network_save_layout(const network* nn, string8 file_name);

string8 network_get_tpn_header(void);

// Saves layer descs and layer params
// *.tpn
void network_save(const network* nn, string8 file_name);

#endif // NETWORK_H

