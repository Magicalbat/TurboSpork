#include "layers.h"
#include "layers_internal.h"

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(prev_shape);

    out->shape = desc->input.shape;
}
void _layer_input_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    UNUSED(cache);

    in_out->shape = l->shape;
}

