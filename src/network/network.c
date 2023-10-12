#include "network.h"

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

    mga_temp scratch = mga_scratch_get(NULL, 0);

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, max_layer_size);
    tensor_copy_ip(in_out, input);

    for (u32 i =0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out);
    }

    tensor_copy_ip(out, in_out);

    mga_scratch_release(scratch);
}
void network_train(network* nn, const network_train_desc* desc) {
}

