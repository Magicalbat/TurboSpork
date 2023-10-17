#include "network.h"

#include <stdio.h>

static u64 _network_max_layer_size(const network* nn);

network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs) {
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    nn->num_layers = num_layers;
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, layer*, nn->num_layers);

    for (u32 i = 0; i < nn->num_layers; i++) {
        nn->layers[i] = layer_create(arena, &layer_descs[i]);
    }

    return nn;
}
void network_feedforward(network* nn, tensor* out, const tensor* input) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    u64 max_layer_size = _network_max_layer_size(nn);

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, max_layer_size);
    tensor_copy_ip(in_out, input);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out);
    }

    tensor_copy_ip(out, in_out);

    mga_scratch_release(scratch);
}
void network_train(network* nn, const network_train_desc* desc) {
    optimizer optim = desc->optim;
    optim._batch_size = desc->batch_size;

    for (u32 epoch = 0; epoch < desc->epochs; epoch++) {
        u32 num_batches = desc->train_inputs->shape.depth / desc->batch_size;

        mga_temp scratch = mga_scratch_get(NULL, 0);

        for (u32 batch = 0; batch < num_batches; batch++) {
            printf("%u / %u\r", batch + 1, num_batches);

            for (u32 i = 0; i < desc->batch_size; i++) {
                u64 index = (u64)i + (u64)batch * desc->batch_size;

                tensor input_view = { 0 };
                tensor output_view = { 0 };
                tensor_2d_view(&input_view, desc->train_inputs, index);
                tensor_2d_view(&output_view, desc->train_outputs, index);

                u64 max_layer_size = _network_max_layer_size(nn);

                tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, max_layer_size);
                tensor_copy_ip(in_out, &input_view);
                tensor* output = tensor_copy(scratch.arena, &output_view, false);

                for (u32 i = 0; i < nn->num_layers; i++) {
                    layer_feedforward(nn->layers[i], in_out);
                }

                // delta is also max_layer_size because of keep_alloc
                tensor* delta = tensor_copy(scratch.arena, in_out, true);
                cost_grad(desc->cost, delta, output);

                for (i64 i = nn->num_layers - 1; i >= 0; i--) {
                    layer_backprop(nn->layers[i], delta);
                }

                // Reset arena
                mga_temp_end(scratch);
            }

            for (u32 i = 0; i < nn->num_layers; i++) {
                layer_apply_changes(nn->layers[i], &optim);
            }
        }

        printf("\n");

        mga_scratch_release(scratch);
    }
}

static u64 _network_max_layer_size(const network* nn) {
    u64 max_layer_size = 0;
    for (u32 i = 0; i < nn->num_layers; i++) {
        tensor_shape s1 = nn->layers[i]->input_shape;
        tensor_shape s2 = nn->layers[i]->output_shape;

        u64 size1 = (u64)s1.width * s1.height * s1.depth;
        u64 size2 = (u64)s2.width * s2.height * s2.depth;

        if (size1 > max_layer_size) {
            max_layer_size = size1;
        }
        if (size2 > max_layer_size) {
            max_layer_size = size2;
        }
    }

    return max_layer_size;
}

