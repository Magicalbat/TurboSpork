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

    // For saving
    layer_desc* layer_descs;

    layer** layers;
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

    tensor* train_inputs;
    tensor* train_outputs;

    b32 accuracy_test;
    tensor* test_inputs;
    tensor* test_outputs;
} network_train_desc;

// This training_mode overrides the one in the desc
network* network_create_new(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode);
// Reads layout file (*.tpl)
// See network_layout_save
network* network_create_layout(mg_arena* arena, string8 file_name, b32 training_mode);
// Reads network file (*.tpn)
// See network_save
network* network_create_existing(mg_arena* arena, string8 file_name, b32 training_mode);

void network_delete(network* nn);

void network_feedforward(const network* nn, tensor* out, const tensor* input);
void network_train(network* nn, const network_train_desc* desc);

// Prints the network summary to stdout
void network_summary(const network* nn);

// Just saves layer descs
// *.tpl
void network_layout_save(const network* nn, string8 file_name);
// Saves layer descs and layer params
// *.tpn
void network_save(const network* nn);

#endif // NETWORK_H

