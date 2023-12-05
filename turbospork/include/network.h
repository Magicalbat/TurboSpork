#ifndef NETWORK_H
#define NETWORK_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"
#include "tensor.h"

#include "layers.h"
#include "costs.h"
#include "optimizers.h"

typedef struct {
    ts_b32 training_mode;

    ts_u32 num_layers;
    ts_layer** layers;

    // For saving
    ts_layer_desc* layer_descs;

    // Used for forward and backward passes
    // Allows for single allocation of input/output variable
    ts_u64 max_layer_size;
} ts_network;

typedef struct {
    ts_u32 epoch;

    ts_f32 test_accuracy;
} ts_network_epoch_info;

typedef void(ts_network_epoch_callback)(const ts_network_epoch_info*);

typedef struct {
    ts_u32 epochs;
    ts_u32 batch_size;

    ts_u32 num_threads;

    ts_cost_type cost;
    ts_optimizer optim;

    // Can be null
    // Gives information to function after each epoch
    ts_network_epoch_callback* epoch_callback;

    // Epoch interval to save network 
    // When (epoch + 1) % save_interval == 0
    // Interval of zero means no saving
    ts_u32 save_interval;
    // Output will be "{save_path}{epoch_num}.tpn"
    ts_string8 save_path;

    ts_tensor* train_inputs;
    ts_tensor* train_outputs;

    ts_b32 accuracy_test;
    ts_tensor* test_inputs;
    ts_tensor* test_outputs;
} ts_network_train_desc;

// This training_mode overrides the one in the desc
ts_network* ts_network_create(mg_arena* arena, ts_u32 num_layers, const ts_layer_desc* layer_descs, ts_b32 training_mode);
// Reads layout file (*.tpl)
// See network_layout_save
ts_network* ts_network_load_layout(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode);
// Reads network file (*.tpn)
// See network_save
ts_network* ts_network_load(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode);

void ts_network_delete(ts_network* nn);

void ts_network_feedforward(const ts_network* nn, ts_tensor* out, const ts_tensor* input);
void ts_network_train(ts_network* nn, const ts_network_train_desc* desc);

// Prints the network summary to stdout
void ts_network_summary(const ts_network* nn);

// Saves layer descs
// *.tpl
void ts_network_save_layout(const ts_network* nn, ts_string8 file_name);

ts_string8 ts_network_get_tpn_header(void);

// Saves layer descs and layer params
// *.tpn
void ts_network_save(const ts_network* nn, ts_string8 file_name);

#endif // NETWORK_H

