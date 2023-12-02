#include "layers.h"
#include "layers_internal.h"

#include <ctype.h>
#include <math.h>
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
    [LAYER_POOLING_2D] = "pooling_2d",
    [LAYER_CONV_2D] = "conv_2d",
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
    [LAYER_POOLING_2D] = {
        _layer_pooling_2d_create,
        _layer_pooling_2d_feedforward,
        _layer_pooling_2d_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,

    },
    [LAYER_CONV_2D] = {
        _layer_conv_2d_create,
        _layer_conv_2d_feedforward,
        _layer_conv_2d_backprop,
        _layer_conv_2d_apply_changes,
        _layer_conv_2d_delete,
        _layer_conv_2d_save,
        _layer_conv_2d_load,
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

static const layer_desc _default_descs[LAYER_COUNT] = {
    [LAYER_NULL] = { },
    [LAYER_INPUT] = { .input.shape = { 1, 1, 1 } },
    [LAYER_DENSE] = {
        .dense = (layer_dense_desc){
            .size = 1,
            .bias_init = PARAM_INIT_ZEROS,
            .weight_init = PARAM_INIT_XAVIER_UNIFORM
        }
    },
    [LAYER_ACTIVATION] = {
        .activation.type = ACTIVATION_RELU
    },
    [LAYER_DROPOUT] = { },
    [LAYER_FLATTEN] = { },
    [LAYER_POOLING_2D] = {
        .pooling_2d = (layer_pooling_2d_desc){
            .pool_size = { 1, 1, 1},
            .type = POOLING_MAX
        }
    },
    [LAYER_CONV_2D] = {
        .conv_2d = (layer_conv_2d_desc) {
            .num_filters = 1,
            .kernel_size = { 1, 1, 1 },
            .stride_x = 1,
            .stride_y = 1,
            .kernels_init = PARAM_INIT_HE_NORMAL,
            .biases_init = PARAM_INIT_ZEROS
        }
    },
};

layer_desc layer_desc_default(layer_type type) {
    if (type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot get default layer desc: invalid layer type\n");

        return (layer_desc){ 0 };
    }

    return _default_descs[type];
}
#define _PARAM_DEFAULT(p, d) (p = p == 0 ? d : p)
layer_desc layer_desc_apply_default(const layer_desc* desc) {
    if (desc->type >= LAYER_COUNT) {
        fprintf(stderr, "Cannot apply defaults to layer desc: invalid layer type\n");

        return (layer_desc){ 0 };
    }

    const layer_desc* def = &_default_descs[desc->type];

    layer_desc out = { 0 };
    memcpy(&out, desc, sizeof(layer_desc));

    switch (desc->type) {
        case LAYER_INPUT: {
            _PARAM_DEFAULT(out.input.shape.width, def->input.shape.width);
            _PARAM_DEFAULT(out.input.shape.height, def->input.shape.height);
            _PARAM_DEFAULT(out.input.shape.depth, def->input.shape.depth);
        } break;
        case LAYER_DENSE: {
            _PARAM_DEFAULT(out.dense.size, def->dense.size);
            _PARAM_DEFAULT(out.dense.bias_init, def->dense.bias_init);
            _PARAM_DEFAULT(out.dense.weight_init, def->dense.weight_init);
        } break;
        case LAYER_ACTIVATION: {
            _PARAM_DEFAULT(out.activation.type, def->activation.type);
        } break;
        case LAYER_DROPOUT: {
            // Nothing yet
        } break;
        case LAYER_POOLING_2D: {
            layer_pooling_2d_desc* out_p = &out.pooling_2d;
            const layer_pooling_2d_desc* def_p = &def->pooling_2d;

            _PARAM_DEFAULT(out_p->pool_size.width, def_p->pool_size.width);
            _PARAM_DEFAULT(out_p->pool_size.height, def_p->pool_size.height);
            _PARAM_DEFAULT(out_p->pool_size.depth, def_p->pool_size.depth);

            _PARAM_DEFAULT(out_p->type, def_p->type);
        } break;
        case LAYER_CONV_2D: {
            layer_conv_2d_desc* out_c = &out.conv_2d;
            const layer_conv_2d_desc* def_c = &def->conv_2d;

            _PARAM_DEFAULT(out_c->num_filters, def_c->num_filters);

            _PARAM_DEFAULT(out_c->kernel_size.width, def_c->kernel_size.width);
            _PARAM_DEFAULT(out_c->kernel_size.height, def_c->kernel_size.height);
            _PARAM_DEFAULT(out_c->kernel_size.depth, def_c->kernel_size.depth);

            _PARAM_DEFAULT(out_c->padding, def_c->padding);
            _PARAM_DEFAULT(out_c->stride_x, def_c->stride_x);
            _PARAM_DEFAULT(out_c->stride_y, def_c->stride_y);
            _PARAM_DEFAULT(out_c->kernels_init, def_c->kernels_init);
            _PARAM_DEFAULT(out_c->biases_init, def_c->biases_init);

        } break;
        default: break;
    }

    return out;
}

static const char* _param_init_names[PARAM_INIT_COUNT] = {
    [PARAM_INIT_NULL] = "null",
    [PARAM_INIT_ZEROS] = "zeros",
    [PARAM_INIT_ONES] = "ones",
    [PARAM_INIT_XAVIER_UNIFORM] = "xavier_uniform",
    [PARAM_INIT_XAVIER_NORMAL] = "xavier_normal",
    [PARAM_INIT_HE_UNIFORM] = "he_uniform",
    [PARAM_INIT_HE_NORMAL] = "he_normal",
};

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

            if (desc->dense.bias_init >= PARAM_INIT_COUNT || desc->dense.weight_init >= PARAM_INIT_COUNT) {
                fprintf(stderr, "Cannot save desc: invalid init type in dense\n");
                break;
            }

            string8 bias_init_str = str8_pushf(
                arena, "    bias_init = %s;\n", _param_init_names[desc->dense.bias_init]
            );
            string8 weight_init_str = str8_pushf(
                arena, "    weight_init = %s;\n", _param_init_names[desc->dense.weight_init]
            );

            str8_list_push(arena, list, size_str);
            str8_list_push(arena, list, bias_init_str);
            str8_list_push(arena, list, weight_init_str);
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
        case LAYER_POOLING_2D: {
            if (desc->pooling_2d.type >= POOLING_COUNT) {
                fprintf(stderr, "Cannot save desc: invalid pooling type\n");

                break;
            }

            string8 type_str = str8_pushf(arena, "    type = %s;\n", _pooling_names[desc->pooling_2d.type]);

            tensor_shape s = desc->pooling_2d.pool_size;
            string8 pool_size_str = str8_pushf(arena, "    pool_size = (%u, %u, %u);\n", s.width, s.height, s.depth);

            str8_list_push(arena, list, type_str);
            str8_list_push(arena, list, pool_size_str);
        } break;
        case LAYER_CONV_2D: {
            const layer_conv_2d_desc* cdesc = &desc->conv_2d;

            if (cdesc->kernels_init >= PARAM_INIT_COUNT || cdesc->biases_init >= PARAM_INIT_COUNT) {
                fprintf(stderr, "Cannot save desc: invalid init type in conv_2d\n");
                break;
            }

            string8 conv_2d_str = str8_pushf(
                arena,
                "   num_filters = %u;\n"
                "   kernel_size = (%u, %u, %u);\n"
                "   padding = %s;\n"
                "   stride_x = %u;\n"
                "   stride_y = %u;\n"
                "   kernels_init = %s;\n"
                "   bias_init = %s;\n",

                cdesc->num_filters,
                cdesc->kernel_size.width, cdesc->kernel_size.height, cdesc->kernel_size.depth, 
                cdesc->padding ? "true" : "false",
                cdesc->stride_x, cdesc->stride_y,
                _param_init_names[cdesc->kernels_init],
                _param_init_names[cdesc->biases_init]
            );

            str8_list_push(arena, list, conv_2d_str);
        }

        default: break;
    }
}

typedef struct {
    b32 error;
    char* err_msg;
} _parse_res;

_parse_res _parse_enum(u32* out, string8 value, const char* enum_names[], u32 num_enums) {
    for (u32 i = 0; i < num_enums; i++) {
        if (str8_equals(value, str8_from_cstr((u8*)enum_names[i]))) {
            *out = i;

            break;
        }
    }

    return (_parse_res){ .error = false };
}
_parse_res _parse_b32(b32* out, string8 value) {
    if (str8_equals(value, STR8("true"))) {
        *out = true;
    } else if (str8_equals(value, STR8("false"))) {
        *out = false;
    } else {
        return (_parse_res) {
            .error = true,
            .err_msg = "Cannot load layer desc: invalid string for bool"
        };
    }

    return (_parse_res){ .error = false };
}
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
_parse_res _parse_u32(u32* out, string8 value) {
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

    u32 num = strtol((char*)num_cstr, &end_ptr, 10);

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

#define _PARSE_RES_ERR_CHECK(res) if (res.error) { \
        fprintf(stderr, "%s\n", res.err_msg); \
        break; \
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

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case LAYER_DENSE: {
                if (str8_equals(key, STR8("size"))) {
                    _parse_res res = _parse_u32(&out.dense.size, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("bias_init"))) {
                    out.dense.bias_init = PARAM_INIT_NULL;
                    u32 bias_init = 0;

                    _parse_res res = _parse_enum(&bias_init, value, _param_init_names, PARAM_INIT_COUNT);
                    out.dense.bias_init = bias_init;

                    if (res.error || out.dense.bias_init == PARAM_INIT_NULL) {
                        fprintf(stderr, "Invalid param init type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                } else if (str8_equals(key, STR8("weight_init"))) {
                    out.dense.weight_init = PARAM_INIT_NULL;
                    u32 weight_init = 0;

                    _parse_res res = _parse_enum(&weight_init, value, _param_init_names, PARAM_INIT_COUNT);
                    out.dense.weight_init = weight_init;

                    if (res.error || out.dense.weight_init == PARAM_INIT_NULL) {
                        fprintf(stderr, "Invalid param init type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                }
            } break;
            case LAYER_ACTIVATION: {
                if (str8_equals(key, STR8("type"))) {
                    out.activation.type = ACTIVATION_NULL;
                    u32 activ_type = 0;

                    _parse_res res = _parse_enum(&activ_type, value, _activ_names, ACTIVATION_COUNT);
                    out.activation.type = activ_type;

                    if (res.error || out.activation.type == ACTIVATION_NULL) {
                        fprintf(stderr, "Invalid activation type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                }
            } break;
            case LAYER_DROPOUT: {
                if (str8_equals(key, STR8("keep_rate"))) {
                    _parse_res res = _parse_f32(&out.dropout.keep_rate, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case LAYER_POOLING_2D: {
                if (str8_equals(key, STR8("type"))) {
                    out.pooling_2d.type = POOLING_NULL;
                    u32 pool_type = 0;

                    _parse_res res = _parse_enum(&pool_type, value, _pooling_names, POOLING_COUNT);
                    out.pooling_2d.type = pool_type;

                    if (res.error || out.pooling_2d.type == POOLING_NULL) {
                        fprintf(stderr, "Invalid pooling type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                } else if (str8_equals(key, STR8("pool_size"))) {
                    _parse_res res = _parse_tensor_shape(&out.pooling_2d.pool_size, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case LAYER_CONV_2D: {
                if (str8_equals(key, STR8("num_filters"))) {
                    _parse_res res = _parse_u32(&out.conv_2d.num_filters, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("kernel_size"))) {
                    _parse_res res = _parse_tensor_shape(&out.conv_2d.kernel_size, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("padding"))) {
                    _parse_res res = _parse_b32(&out.conv_2d.padding, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("stride_x"))) {
                    _parse_res res = _parse_u32(&out.conv_2d.stride_x, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("stride_y"))) {
                    _parse_res res = _parse_u32(&out.conv_2d.stride_x, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (str8_equals(key, STR8("kernels_init"))) {
                    out.conv_2d.kernels_init = PARAM_INIT_NULL;
                    u32 kernels_init = 0;

                    _parse_res res = _parse_enum(&kernels_init, value, _param_init_names, PARAM_INIT_COUNT);
                    out.conv_2d.kernels_init = kernels_init;

                    if (res.error || out.dense.bias_init == PARAM_INIT_NULL) {
                        fprintf(stderr, "Invalid param init type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                } else if (str8_equals(key, STR8("biases_init"))) {
                    out.conv_2d.biases_init = PARAM_INIT_NULL;
                    u32 biases_init = 0;

                    _parse_res res = _parse_enum(&biases_init, value, _param_init_names, PARAM_INIT_COUNT);
                    out.conv_2d.biases_init = biases_init;

                    if (res.error || out.dense.bias_init == PARAM_INIT_NULL) {
                        fprintf(stderr, "Invalid param init type \"%.*s\"\n", (int)value.size, (char*)value.str);
                    }
                }
            }
            default: break;
        }
    }

    mga_scratch_release(scratch);

    return out;
}

void _param_init_uniform(tensor* param, f32 lower, f32 upper) {
    u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
    for (u64 i = 0; i < size; i++) {
        param->data[i] = prng_rand_f32();
        param->data[i] = param->data[i] * (upper - lower) - lower;
    }
}

void _param_init_normal(tensor* param, f32 mean, f32 std_dev) {
    u64 size = (u64)param->shape.width * param->shape.height * param->shape.depth;
    for (u64 i = 0; i < size; i++) {
        param->data[i] = prng_std_norm();
        param->data[i] = mean + param->data[i] * std_dev;
    }
}

void param_init(tensor* param, param_init_type input_type, u64 in_size, u64 out_size) {
    switch (input_type) {
        case PARAM_INIT_ZEROS: {
            tensor_fill(param, 0.0f);
        } break;
        case PARAM_INIT_ONES: {
            tensor_fill(param, 1.0f);
        } break;
        case PARAM_INIT_XAVIER_UNIFORM: {
            f32 upper = sqrtf(6.0f / (f32)(in_size + out_size));
            _param_init_uniform(param, -upper, upper);
        } break;
        case PARAM_INIT_XAVIER_NORMAL: {
            f32 std_dev = sqrtf(2.0f / (f32)(in_size + out_size));
            _param_init_normal(param, 0.0f, std_dev);
        } break;
        case PARAM_INIT_HE_UNIFORM: {
            f32 upper = sqrtf(6.0f / (f32)in_size);
            _param_init_uniform(param, -upper, upper);
        }
        case PARAM_INIT_HE_NORMAL: {
            f32 std_dev = sqrtf(2.0f / (f32)in_size);
            _param_init_normal(param, 0.0f, std_dev);
        }
        default: break;
    }
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

