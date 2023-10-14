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
    u32 epochs;
    u32 batch_size;

    cost_type cost;
    optimizer_desc optim_desc;

    tensor* train_inputs;
    tensor* train_outputs;

    b32 accuracy_test;
    tensor* test_inputs;
    tensor* test_outputs;
} network_train_desc;

network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs);
void network_feedforward(network* nn, tensor* out, const tensor* input);
void network_train(network* nn, const network_train_desc* desc);

#endif // NETWORK_H

