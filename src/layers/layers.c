#include "layers.h"
#include "layers_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* _layer_names[LAYER_COUNT] = {
    [LAYER_NULL] = "null",
    [LAYER_INPUT] = "input",
    [LAYER_DENSE] = "dense",
    [LAYER_ACTIVATION] = "activation",
    [LAYER_DROPOUT] = "dropout",
    [LAYER_FLATTEN] = "flatten",
    [LAYER_POOLING] = "pooling"
};

string8 layer_get_name(layer_type type) {
    if (type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot get name: invalid layer type\n");

        return (string8){ 0 };
    }

    return str8_from_cstr((u8*)_layer_names[type]);
}
layer_type layer_from_name(string8 name) {
    for (layer_type type = LAYER_NULL; type < LAYER_COUNT; type++) {
        if (str8_equals(name, layer_get_name(type))) {
            return type;
        }
    }

    return LAYER_NULL;
}

static _layer_func_defs _layer_funcs[LAYER_COUNT] = {
    [LAYER_NULL] = {
        _layer_null_create,
        _layer_null_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [LAYER_INPUT] = {
        _layer_input_create,
        _layer_null_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [LAYER_DENSE] = {
        _layer_dense_create,
        _layer_dense_feedforward,
        _layer_dense_backprop,
        _layer_dense_apply_changes,
        _layer_dense_delete,
        _layer_dense_save,
        _layer_dense_load,
    },
    [LAYER_ACTIVATION] = {
        _layer_activation_create,
        _layer_activation_feedforward,
        _layer_activation_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [LAYER_DROPOUT] = {
        _layer_dropout_create,
        _layer_dropout_feedforward,
        _layer_dropout_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [LAYER_FLATTEN] = {
        _layer_flatten_create,
        _layer_flatten_feedforward,
        _layer_flatten_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,

    },
    [LAYER_POOLING] = {
        _layer_pooling_create,
        _layer_pooling_feedforward,
        _layer_pooling_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,

    }
};

#define _TYPE_CHECK(err_msg) do { \
        if (l->type >= LAYER_COUNT) { \
            fprintf(stderr, err_msg); \
            return; \
        } \
    } while (0);

layer* layer_create(mg_arena* arena, const layer_desc* desc, tensor_shape prev_shape) {
    if (desc->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot create layer: invalid type\n");
        return NULL;
    }

    layer* out = MGA_PUSH_ZERO_STRUCT(arena, layer);

    out->type = desc->type;
    out->training_mode = desc->training_mode;

    _layer_funcs[desc->type].create(arena, out, desc, prev_shape);

    return out;
}
void layer_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    _TYPE_CHECK("Cannot feedforward layer: invalid type\n");

    _layer_funcs[l->type].feedforward(l, in_out, cache);
}
void layer_backprop(layer* l, tensor* delta, layers_cache* cache) {
    _TYPE_CHECK("Cannot backprop layer: invalid type\n");

    _layer_funcs[l->type].backprop(l, delta, cache);
}
void layer_apply_changes(layer* l, const optimizer* optim) {
    _TYPE_CHECK("Cannot apply changes in layer: invalid type\n");

    _layer_funcs[l->type].apply_changes(l, optim);
}
void layer_delete(layer* l) {
    _TYPE_CHECK("Cannot delete layer: invalid type\n");

    _layer_funcs[l->type].delete(l);
}
void layer_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {
    _TYPE_CHECK("Cannot save layer: invalid type\n");

    _layer_funcs[l->type].save(arena, list, l, index);
}
void layer_load(layer* l, const tensor_list* list, u32 index) {
    _TYPE_CHECK("Cannot load layer: invalid type\n");

    _layer_funcs[l->type].load(l, list, index);
}

static const char* _activ_names[ACTIVATION_COUNT] = {
    [ACTIVATION_NULL] = "null",
    [ACTIVATION_SIGMOID] = "sigmoid",
    [ACTIVATION_TANH] = "tanh",
    [ACTIVATION_RELU] = "relu",
    [ACTIVATION_LEAKY_RELU] = "leaky_relu",
    [ACTIVATION_SOFTMAX] = "softmax",
};

static const char* _pooling_names[POOLING_COUNT] = {
    [POOLING_NULL] = "null",
    [POOLING_AVG] = "average",
    [POOLING_MAX] = "max",
    [POOLING_L2] = "l2"
};

/*
Layer desc save:

layer_type:
    field = value;
    field = value;

*/
void layer_desc_save(mg_arena* arena, string8_list* list, const layer_desc* desc) {
    if (desc->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot save desc: invalid layer type\n");

        return;
    }

    string8 type_str = layer_get_name(desc->type);

    // type:\n
    u64 out_type_size = type_str.size + 2;
    string8 out_type = {
        .size = out_type_size,
        .str = MGA_PUSH_ZERO_ARRAY(arena, u8, out_type_size)
    };
    out_type.str[out_type_size - 2] = ':';
    out_type.str[out_type_size - 1] = '\n';
    memcpy(out_type.str, type_str.str, type_str.size);

    str8_list_push(arena, list, out_type);

    switch (desc->type) {
        case LAYER_INPUT: {
            tensor_shape s = desc->input.shape;
            string8 shape_str = str8_pushf(arena, "    shape = (%u, %u, %u);\n", s.width, s.height, s.depth);

            str8_list_push(arena, list, shape_str);
        } break;
        case LAYER_DENSE: {
            string8 size_str = str8_pushf(arena, "    size = %u;\n", desc->dense.size);

            str8_list_push(arena, list, size_str);
        } break;
        case LAYER_ACTIVATION: {
            if (desc->activation.type >= ACTIVATION_COUNT) {
                fprintf(stderr, "Cannot save desc: invalid activation type\n");

                break;
            }

            string8 type_str = str8_pushf(arena, "    type = %s;\n", _activ_names[desc->activation.type]);

            str8_list_push(arena, list, type_str);
        } break;
        case LAYER_DROPOUT: {
            string8 rate_str = str8_pushf(arena, "    keep_rate = %f;\n", desc->dropout.keep_rate);

            str8_list_push(arena, list, rate_str);
        } break;
        case LAYER_POOLING: {
            if (desc->pooling.type >= POOLING_COUNT) {
                fprintf(stderr, "Cannot save desc: invalid pooling type\n");

                break;
            }

            string8 type_str = str8_pushf(arena, "    type = %s;\n", _pooling_names[desc->pooling.type]);

            tensor_shape s = desc->pooling.pool_size;
            string8 pool_size_str = str8_pushf(arena, "    pool_size = (%u, %u, %u);\n", s.width, s.height, s.depth);

            str8_list_push(arena, list, type_str);
            str8_list_push(arena, list, pool_size_str);
        }

        default: break;
    }
}

typedef struct {
    b32 error;
    char* err_msg;
} _parse_res;

_parse_res _parse_tensor_shape(tensor_shape* out, string8 value) {
    // Parsing (w, h, d)

    // Comma indices
    u64 comma_1 = 0;
    u64 comma_2 = 0;

    b32 valid = true;

    if (value.str[0] != '(' || value.str[value.size - 1] != ')') {
        valid = false;
    }

    for (u64 i = 1; i < value.size - 1; i++) {
        if (value.str[i] == ',') {
            if (comma_1 == 0) {
                comma_1 = i;
            } else {
                comma_2 = i;
            }
        } else if (!isdigit(value.str[i])) {
            valid = false;
            break;
        }
    }

    if (comma_1 == 0 || comma_2 == 0) {
        valid = false;
    }

    if (!valid) {
        return (_parse_res){ 
            .error = true,
            .err_msg = "Cannot load layer desc: Invalid tensor shape format, must be format \"(w,h,d)\""
        };
    }

    // Getting w, h, and d strings
    string8 num1_str = str8_substr(value, 1, comma_1);
    string8 num2_str = str8_substr(value, comma_1 + 1, comma_2);
    string8 num3_str = str8_substr(value, comma_2 + 1, value.size-1);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // strtol requires c strings
    u8* num1_cstr = str8_to_cstr(scratch.arena, num1_str);
    u8* num2_cstr = str8_to_cstr(scratch.arena, num2_str);
    u8* num3_cstr = str8_to_cstr(scratch.arena, num3_str);

    tensor_shape shape = {};
    char* end_ptr = NULL;

    shape.width = strtol((char*)num1_cstr, &end_ptr, 10);
    shape.height = strtol((char*)num2_cstr, &end_ptr, 10);
    shape.depth = strtol((char*)num3_cstr, &end_ptr, 10);

    *out = shape;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}
_parse_res _parse_u64(u64* out, string8 value) {
    // Checking if the string is made of valid characters
    b32 is_num = true;

    for (u64 i = 0; i < value.size; i++) {
        if (!isdigit(value.str[i])) {
            is_num = false;
            break;
        }
    }

    if (!is_num) {
        return (_parse_res) {
            .error = true,
            .err_msg = "Cannot load layer desc: invalid character for number"
        };
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // strtol requiers c strings
    u8* num_cstr = str8_to_cstr(scratch.arena, value);
    char* end_ptr = NULL;

    u64 num = strtoll((char*)num_cstr, &end_ptr, 10);

    *out = num;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}
_parse_res _parse_f32(f32* out, string8 value) {
    // Checking if the string is made of valid characters
    b32 is_num = true;

    for (u64 i = 0; i < value.size; i++) {
        if (!isdigit(value.str[i]) && value.str[i] != '.' && value.str[i] != '-') {
            is_num = false;
            break;
        }
    }

    if (!is_num) {
        return (_parse_res) {
            .error = true,
            .err_msg = "Cannot load layer desc: invalid character for number"
        };
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // strtol requiers c strings
    u8* num_cstr = str8_to_cstr(scratch.arena, value);
    char* end_ptr = NULL;

    f32 num = strtof((char*)num_cstr, &end_ptr);

    *out = num;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}

layer_desc layer_desc_load(string8 str) {
    layer_desc out = { };

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 stripped_str = str8_remove_space(scratch.arena, str);

    u64 colon_index = 0;
    if (!str8_index_of_char(stripped_str, (u8)':', &colon_index)) {
        fprintf(stderr, "Cannot load layer desc: Invalid string\n");

        mga_scratch_release(scratch);
    }

    string8 type_str = str8_substr(stripped_str, 0, colon_index);

    out.type = layer_from_name(type_str);

    string8 cur_str = str8_substr(stripped_str, type_str.size + 1, stripped_str.size + 1);

    while (cur_str.size > 0) {
        // Parsing of format
        // key=value;

        u64 eq_index = 0;
        if (!str8_index_of_char(cur_str, (u8)'=', &eq_index)) {
            fprintf(stderr, "Cannot load layer desc: Invalid field (no '=')\n");
            break;
        }
        
        u64 semi_index = 0;
        if (!str8_index_of_char(cur_str, (u8)';', &semi_index)) {
            fprintf(stderr, "Cannot load layer desc: Invalid field (no ';')\n");
            break;
        }

        string8 key = str8_substr(cur_str, 0, eq_index);
        string8 value = str8_substr(cur_str, eq_index + 1, semi_index);

        cur_str.str += semi_index + 1;
        cur_str.size -= semi_index + 1;

        if (key.size == 0 || value.size == 0) {
            fprintf(stderr, "Cannot load layer desc: Invalid key/value\n");

            break;
        }

        switch (out.type) {
            case LAYER_INPUT: {
                if (str8_equals(key, STR8("shape"))) {
                    _parse_res res = _parse_tensor_shape(&out.input.shape, value);

                    if (res.error) {
                        fprintf(stderr, "%s\n", res.err_msg);

                        break;
                    }
                }
            } break;
            case LAYER_DENSE: {
                if (str8_equals(key, STR8("size"))) {
                    u64 size = 0;

                    _parse_res res = _parse_u64(&size, value);

                    out.dense.size = (u32)size;

                    if (res.error) {
                        fprintf(stderr, "%s\n", res.err_msg);

                        break;
                    }
                }
            } break;
            case LAYER_ACTIVATION: {
                if (str8_equals(key, STR8("type"))) {
                    // Type is an enum
                    // Strings of each enum are in `_activ_names`

                    for (u32 i = 0; i < ACTIVATION_COUNT; i++) {
                        if (str8_equals(value, str8_from_cstr((u8*)_activ_names[i]))) {
                            out.activation.type = i;

                            break;
                        }
                    }
                }
            } break;
            case LAYER_DROPOUT: {
                if (str8_equals(key, STR8("keep_rate"))) {
                    _parse_res res = _parse_f32(&out.dropout.keep_rate, value);

                    if (res.error) {
                        fprintf(stderr, "%s\n", res.err_msg);

                        break;
                    }
                }
            } break;
            case LAYER_POOLING: {
                if (str8_equals(key, STR8("type"))) {
                    // Type is an enum
                    // Strings of each enum are in `_pooling_names`

                    for (u32 i = 0; i < POOLING_COUNT; i++) {
                        if (str8_equals(value, str8_from_cstr((u8*)_pooling_names[i]))) {
                            out.pooling.type = i;

                            break;
                        }
                    }
                } else if (str8_equals(key, STR8("pool_size"))) {
                    _parse_res res = _parse_tensor_shape(&out.pooling.pool_size, value);

                    if (res.error) {
                        fprintf(stderr, "%s\n", res.err_msg);

                        break;
                    }
                }
            } break;
            default: break;
        }
    }

    mga_scratch_release(scratch);

    return out;
}

void layers_cache_push(layers_cache* cache, tensor* t) {
    layers_cache_node* node = MGA_PUSH_ZERO_STRUCT(cache->arena, layers_cache_node);
    node->t = t;

    SLL_PUSH_FRONT(cache->first, cache->last, node);
}
tensor* layers_cache_pop(layers_cache* cache) {
    tensor* out = cache->first->t;

    SLL_POP_FRONT(cache->first, cache->last);

    return out;
}

void _layer_null_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(desc);

    out->shape = prev_shape;
}
void _layer_null_feedforward(layer* l, tensor* in_out, layers_cache* cache) {
    UNUSED(l);
    UNUSED(in_out);
    UNUSED(cache);
}
void _layer_null_backprop(layer* l, tensor* delta, layers_cache* cache) {
    UNUSED(l);
    UNUSED(delta);
    UNUSED(cache);
}
void _layer_null_apply_changes(layer* l, const optimizer* optim) {
    UNUSED(l);
    UNUSED(optim);
}
void _layer_null_delete(layer* l) {
    UNUSED(l);
}
void _layer_null_save(mg_arena* arena, tensor_list* list, layer* l, u32 index) {
    UNUSED(arena);
    UNUSED(list);
    UNUSED(l);
    UNUSED(index);
}
void _layer_null_load(layer* l, const tensor_list* list, u32 index) {
    UNUSED(l);
    UNUSED(list);
    UNUSED(index);
}

void _layer_input_create(mg_arena* arena, layer* out, const layer_desc* desc, tensor_shape prev_shape) {
    UNUSED(arena);
    UNUSED(prev_shape);

    out->shape = desc->input.shape;

    // Input layer never needs to be in training mode
    out->training_mode = false;
}

