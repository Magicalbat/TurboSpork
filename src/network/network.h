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
network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode);
void network_delete(network* nn);

void network_feedforward(network* nn, tensor* out, const tensor* input);
void network_train(network* nn, const network_train_desc* desc);

// Prints the network summary to stdout
void network_summary(const network* nn);

#endif // NETWORK_H

