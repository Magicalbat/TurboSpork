#include "layers.h"
#include "err.h"
#include "layers_internal.h"
#include "prng.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* _layer_names[TS_LAYER_COUNT] = {
    [TS_LAYER_NULL] = "null",
    [TS_LAYER_INPUT] = "input",
    [TS_LAYER_RESHAPE] = "reshape",
    [TS_LAYER_DENSE] = "dense",
    [TS_LAYER_ACTIVATION] = "activation",
    [TS_LAYER_DROPOUT] = "dropout",
    [TS_LAYER_FLATTEN] = "flatten",
    [TS_LAYER_POOLING_2D] = "pooling_2d",
    [TS_LAYER_CONV_2D] = "conv_2d",
    [TS_LAYER_NORM] = "norm"
};

ts_string8 ts_layer_get_name(ts_layer_type type) {
    if (type >= TS_LAYER_COUNT) {
        fprintf(stderr, "Cannot get name: invalid layer type\n");

        return (ts_string8){ 0 };
    }

    return ts_str8_from_cstr((ts_u8*)_layer_names[type]);
}
ts_layer_type ts_layer_from_name(ts_string8 name) {
    for (ts_layer_type type = TS_LAYER_NULL; type < TS_LAYER_COUNT; type++) {
        if (ts_str8_equals(name, ts_layer_get_name(type))) {
            return type;
        }
    }

    return TS_LAYER_NULL;
}

static _layer_func_defs _layer_funcs[TS_LAYER_COUNT] = {
    [TS_LAYER_NULL] = {
        _layer_null_create,
        _layer_null_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [TS_LAYER_INPUT] = {
        _layer_input_create,
        _layer_input_feedforward,
        _layer_null_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [TS_LAYER_RESHAPE] = {
        _layer_reshape_create,
        _layer_reshape_feedforward,
        _layer_reshape_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [TS_LAYER_DENSE] = {
        _layer_dense_create,
        _layer_dense_feedforward,
        _layer_dense_backprop,
        _layer_dense_apply_changes,
        _layer_dense_delete,
        _layer_dense_save,
        _layer_dense_load,
    },
    [TS_LAYER_ACTIVATION] = {
        _layer_activation_create,
        _layer_activation_feedforward,
        _layer_activation_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [TS_LAYER_DROPOUT] = {
        _layer_dropout_create,
        _layer_dropout_feedforward,
        _layer_dropout_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    },
    [TS_LAYER_FLATTEN] = {
        _layer_flatten_create,
        _layer_flatten_feedforward,
        _layer_flatten_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,

    },
    [TS_LAYER_POOLING_2D] = {
        _layer_pooling_2d_create,
        _layer_pooling_2d_feedforward,
        _layer_pooling_2d_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,

    },
    [TS_LAYER_CONV_2D] = {
        _layer_conv_2d_create,
        _layer_conv_2d_feedforward,
        _layer_conv_2d_backprop,
        _layer_conv_2d_apply_changes,
        _layer_conv_2d_delete,
        _layer_conv_2d_save,
        _layer_conv_2d_load,
    },
    [TS_LAYER_NORM] = {
        _layer_norm_create,
        _layer_norm_feedforward,
        _layer_norm_backprop,
        _layer_null_apply_changes,
        _layer_null_delete,
        _layer_null_save,
        _layer_null_load,
    }
};

#define _ERR_CHECK(err_msg) do { \
        if (l == NULL) { \
            TS_ERR(TS_ERR_INVALID_INPUT, err_msg ": layer is NULL"); \
            return; \
        } \
        if (l->type >= TS_LAYER_COUNT) { \
            TS_ERR(TS_ERR_INVALID_ENUM, err_msg ": invalid layer type"); \
            return; \
        } \
    } while (0);

ts_layer* ts_layer_create(mg_arena* arena, const ts_layer_desc* desc, ts_tensor_shape prev_shape) {
    if (desc == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create layer with NULL desc");
        return NULL;
    }
    if (desc->type >= TS_LAYER_COUNT) {
        TS_ERR(TS_ERR_INVALID_ENUM, "Cannot create layer: invalid type");
        return NULL;
    }

    ts_layer* out = MGA_PUSH_ZERO_STRUCT(arena, ts_layer);

    out->type = desc->type;
    out->training_mode = desc->training_mode;

    _layer_funcs[desc->type].create(arena, out, desc, prev_shape);

    return out;
}
void ts_layer_feedforward(ts_layer* l, ts_tensor* in_out, ts_layers_cache* cache) {
    _ERR_CHECK("Cannot feedforward layer");

    _layer_funcs[l->type].feedforward(l, in_out, cache);
}
void ts_layer_backprop(ts_layer* l, ts_tensor* delta, ts_layers_cache* cache) {
    _ERR_CHECK("Cannot backprop layer");

    _layer_funcs[l->type].backprop(l, delta, cache);
}
void ts_layer_apply_changes(ts_layer* l, const ts_optimizer* optim) {
    _ERR_CHECK("Cannot apply changes in layer");

    _layer_funcs[l->type].apply_changes(l, optim);
}
void ts_layer_delete(ts_layer* l) {
    _ERR_CHECK("Cannot delete layer");

    _layer_funcs[l->type].delete(l);
}
void ts_layer_save(mg_arena* arena, ts_layer* l, ts_tensor_list* list, ts_u32 index) {
    _ERR_CHECK("Cannot save layer");

    _layer_funcs[l->type].save(arena, l, list, index);
}
void ts_layer_load(ts_layer* l, const ts_tensor_list* list, ts_u32 index) {
    _ERR_CHECK("Cannot load layer");

    _layer_funcs[l->type].load(l, list, index);
}

static const ts_layer_desc _default_descs[TS_LAYER_COUNT] = {
    [TS_LAYER_NULL] = { 0 },
    [TS_LAYER_INPUT] = { 0 },
    [TS_LAYER_RESHAPE] = { 0 },
    [TS_LAYER_DENSE] = {
        .dense = {
            .bias_init = TS_PARAM_INIT_ZEROS,
            .weight_init = TS_PARAM_INIT_XAVIER_UNIFORM
        }
    },
    [TS_LAYER_ACTIVATION] = {
        .activation.type = TS_ACTIVATION_RELU
    },
    [TS_LAYER_DROPOUT] = { 0 },
    [TS_LAYER_FLATTEN] = { 0 },
    [TS_LAYER_POOLING_2D] = {
        .pooling_2d = {
            .type = TS_POOLING_MAX
        }
    },
    [TS_LAYER_CONV_2D] = {
        .conv_2d = {
            .stride = 1,
            .kernels_init = TS_PARAM_INIT_HE_NORMAL,
            .biases_init = TS_PARAM_INIT_ZEROS
        }
    },
    [TS_LAYER_NORM] = {
        .norm = {
            .epsilon = 1e-5
        }
    }
};

ts_layer_desc ts_layer_desc_default(ts_layer_type type) {
    if (type >= TS_LAYER_COUNT) {
        TS_ERR(TS_ERR_INVALID_ENUM, "Cannot get default layer desc: invalid layer type");

        return (ts_layer_desc){ 0 };
    }

    return _default_descs[type];
}
#define _PARAM_DEFAULT(p, d) (p = p == 0 ? d : p)
ts_layer_desc ts_layer_desc_apply_default(const ts_layer_desc* desc) {
    if (desc == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot apply defaults of NULL desc");

        return (ts_layer_desc){ 0 };
    }
    if (desc->type >= TS_LAYER_COUNT) {
        TS_ERR(TS_ERR_INVALID_ENUM, "Cannot apply defaults to layer desc: invalid layer type");

        return (ts_layer_desc){ 0 };
    }

    const ts_layer_desc* def = &_default_descs[desc->type];

    ts_layer_desc out = { 0 };
    memcpy(&out, desc, sizeof(ts_layer_desc));

    switch (desc->type) {
        case TS_LAYER_DENSE: {
            _PARAM_DEFAULT(out.dense.bias_init, def->dense.bias_init);
            _PARAM_DEFAULT(out.dense.weight_init, def->dense.weight_init);
        } break;
        case TS_LAYER_ACTIVATION: {
            _PARAM_DEFAULT(out.activation.type, def->activation.type);
        } break;
        case TS_LAYER_POOLING_2D: {
            ts_layer_pooling_2d_desc* out_p = &out.pooling_2d;
            const ts_layer_pooling_2d_desc* def_p = &def->pooling_2d;

            _PARAM_DEFAULT(out_p->type, def_p->type);
        } break;
        case TS_LAYER_CONV_2D: {
            ts_layer_conv_2d_desc* out_c = &out.conv_2d;
            const ts_layer_conv_2d_desc* def_c = &def->conv_2d;

            _PARAM_DEFAULT(out_c->stride, def_c->stride);

            _PARAM_DEFAULT(out_c->kernels_init, def_c->kernels_init);
            _PARAM_DEFAULT(out_c->biases_init, def_c->biases_init);

        } break;
        case TS_LAYER_NORM: {
            _PARAM_DEFAULT(out.norm.epsilon, def->norm.epsilon);
        } break;
        default: break;
    }

    return out;
}

static const char* _param_init_names[TS_PARAM_INIT_COUNT] = {
    [TS_PARAM_INIT_NULL] = "null",
    [TS_PARAM_INIT_ZEROS] = "zeros",
    [TS_PARAM_INIT_ONES] = "ones",
    [TS_PARAM_INIT_XAVIER_UNIFORM] = "xavier_uniform",
    [TS_PARAM_INIT_XAVIER_NORMAL] = "xavier_normal",
    [TS_PARAM_INIT_HE_UNIFORM] = "he_uniform",
    [TS_PARAM_INIT_HE_NORMAL] = "he_normal",
};

static const char* _activ_names[TS_ACTIVATION_COUNT] = {
    [TS_ACTIVATION_NULL] = "null",
    [TS_ACTIVATION_LINEAR] = "linear",
    [TS_ACTIVATION_SIGMOID] = "sigmoid",
    [TS_ACTIVATION_TANH] = "tanh",
    [TS_ACTIVATION_RELU] = "relu",
    [TS_ACTIVATION_LEAKY_RELU] = "leaky_relu",
    [TS_ACTIVATION_SOFTMAX] = "softmax",
};

static const char* _pooling_names[TS_POOLING_COUNT] = {
    [TS_POOLING_NULL] = "null",
    [TS_POOLING_AVG] = "average",
    [TS_POOLING_MAX] = "max",
};

/*
Layer desc save:

layer_type:
    field = value;
    field = value;

*/
void ts_layer_desc_save(mg_arena* arena, ts_string8_list* list, const ts_layer_desc* desc) {
    if (list == NULL || desc == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot save desc: list or desc is NULL");
    }
    if (desc->type >= TS_LAYER_COUNT) {
        TS_ERR(TS_ERR_INVALID_ENUM, "Cannot save desc: invalid layer type");

        return;
    }

    ts_string8 type_str = ts_layer_get_name(desc->type);

    // type:\n
    ts_u64 out_type_size = type_str.size + 2;
    ts_string8 out_type = {
        .size = out_type_size,
        .str = MGA_PUSH_ZERO_ARRAY(arena, ts_u8, out_type_size)
    };
    out_type.str[out_type_size - 2] = ':';
    out_type.str[out_type_size - 1] = '\n';
    memcpy(out_type.str, type_str.str, type_str.size);

    ts_str8_list_push(arena, list, out_type);

    switch (desc->type) {
        case TS_LAYER_INPUT: {
            ts_tensor_shape s = desc->input.shape;
            ts_string8 shape_str = ts_str8_pushf(arena, "    shape = (%u, %u, %u);\n", s.width, s.height, s.depth);

            ts_str8_list_push(arena, list, shape_str);
        } break;
        case TS_LAYER_RESHAPE: {
            ts_tensor_shape s = desc->reshape.shape;
            ts_string8 shape_str = ts_str8_pushf(arena, "    shape = (%u, %u, %u);\n", s.width, s.height, s.depth);

            ts_str8_list_push(arena, list, shape_str);
        } break;
        case TS_LAYER_DENSE: {
            ts_string8 size_str = ts_str8_pushf(arena, "    size = %u;\n", desc->dense.size);

            if (desc->dense.bias_init >= TS_PARAM_INIT_COUNT || desc->dense.weight_init >= TS_PARAM_INIT_COUNT) {
                TS_ERR(TS_ERR_INVALID_ENUM, "Cannot save desc: invalid init type in dense");
                break;
            }

            ts_string8 bias_init_str = ts_str8_pushf(
                arena, "    bias_init = %s;\n", _param_init_names[desc->dense.bias_init]
            );
            ts_string8 weight_init_str = ts_str8_pushf(
                arena, "    weight_init = %s;\n", _param_init_names[desc->dense.weight_init]
            );

            ts_str8_list_push(arena, list, size_str);
            ts_str8_list_push(arena, list, bias_init_str);
            ts_str8_list_push(arena, list, weight_init_str);
        } break;
        case TS_LAYER_ACTIVATION: {
            if (desc->activation.type >= TS_ACTIVATION_COUNT) {
                TS_ERR(TS_ERR_INVALID_ENUM, "Cannot save desc: invalid activation type");
                break;
            }

            ts_string8 type_str = ts_str8_pushf(arena, "    type = %s;\n", _activ_names[desc->activation.type]);

            ts_str8_list_push(arena, list, type_str);
        } break;
        case TS_LAYER_DROPOUT: {
            ts_string8 rate_str = ts_str8_pushf(arena, "    keep_rate = %f;\n", desc->dropout.keep_rate);

            ts_str8_list_push(arena, list, rate_str);
        } break;
        case TS_LAYER_POOLING_2D: {
            if (desc->pooling_2d.type >= TS_POOLING_COUNT) {
                TS_ERR(TS_ERR_INVALID_ENUM, "Cannot save desc: invalid pooling type");
                break;
            }

            ts_string8 type_str = ts_str8_pushf(arena, "    type = %s;\n", _pooling_names[desc->pooling_2d.type]);

            ts_tensor_shape s = desc->pooling_2d.pool_size;
            ts_string8 pool_size_str = ts_str8_pushf(arena, "    pool_size = (%u, %u);\n", s.width, s.height);

            ts_str8_list_push(arena, list, type_str);
            ts_str8_list_push(arena, list, pool_size_str);
        } break;
        case TS_LAYER_CONV_2D: {
            const ts_layer_conv_2d_desc* cdesc = &desc->conv_2d;

            if (cdesc->kernels_init >= TS_PARAM_INIT_COUNT || cdesc->biases_init >= TS_PARAM_INIT_COUNT) {
                TS_ERR(TS_ERR_INVALID_ENUM, "Cannot save desc: invalid init type in conv_2d");
                break;
            }

            ts_string8 conv_2d_str = ts_str8_pushf(
                arena,
                "   num_filters = %u;\n"
                "   kernel_size = %u;\n"
                "   stride = %u;\n"
                "   padding = %s;\n"
                "   kernels_init = %s;\n"
                "   bias_init = %s;\n",

                cdesc->num_filters,
                cdesc->kernel_size, 
                cdesc->stride,
                cdesc->padding ? "true" : "false",
                _param_init_names[cdesc->kernels_init],
                _param_init_names[cdesc->biases_init]
            );

            ts_str8_list_push(arena, list, conv_2d_str);
        } break;
        case TS_LAYER_NORM: {
            ts_string8 epsilon_str = ts_str8_pushf(arena, "    epsilon = %e;\n", desc->norm.epsilon);

            ts_str8_list_push(arena, list, epsilon_str);
        } break;

        default: break;
    }
}

typedef struct {
    ts_b32 error;
    char* err_msg;
} _parse_res;

_parse_res _parse_enum(ts_u32* out, ts_string8 value, const char* enum_names[], ts_u32 num_enums) {
    for (ts_u32 i = 0; i < num_enums; i++) {
        if (ts_str8_equals(value, ts_str8_from_cstr((ts_u8*)enum_names[i]))) {
            *out = i;

            break;
        }
    }

    return (_parse_res){ .error = false };
}
_parse_res _parse_b32(ts_b32* out, ts_string8 value) {
    if (ts_str8_equals(value, TS_STR8("true"))) {
        *out = true;
    } else if (ts_str8_equals(value, TS_STR8("false"))) {
        *out = false;
    } else {
        return (_parse_res) {
            .error = true,
            .err_msg = "Cannot load layer desc: invalid string for bool"
        };
    }

    return (_parse_res){ .error = false };
}
_parse_res _parse_tensor_shape(ts_tensor_shape* out, ts_string8 value) {
    // Parsing (w, h) or (w, h, d)

    // Comma indices
    ts_u64 comma_1 = 0;
    ts_u64 comma_2 = 0;

    ts_b32 valid = true;

    if (value.str[0] != '(' || value.str[value.size - 1] != ')') {
        valid = false;
    }

    for (ts_u64 i = 1; i < value.size - 1; i++) {
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

    if (comma_1 == 0) {
        valid = false;
    }

    if (!valid) {
        return (_parse_res){ 
            .error = true,
            .err_msg = "Cannot load layer desc: Invalid ts_tensor shape format, must be format \"(w,h,d)\""
        };
    }

    ts_b32 is_2d = false;
    if (comma_2 == 0) {
        is_2d = true;
        comma_2 = value.size - 1;
    }

    // Getting w, h, and d strings
    ts_string8 num1_str = ts_str8_substr(value, 1, comma_1);
    ts_string8 num2_str = ts_str8_substr(value, comma_1 + 1, comma_2);
    ts_string8 num3_str = ts_str8_substr(value, comma_2 + 1, value.size-1);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // strtol requires c strings
    ts_u8* num1_cstr = ts_str8_to_cstr(scratch.arena, num1_str);
    ts_u8* num2_cstr = ts_str8_to_cstr(scratch.arena, num2_str);
    ts_u8* num3_cstr = ts_str8_to_cstr(scratch.arena, num3_str);

    ts_tensor_shape shape = { 0 };
    char* end_ptr = NULL;

    shape.width = strtol((char*)num1_cstr, &end_ptr, 10);
    shape.height = strtol((char*)num2_cstr, &end_ptr, 10);

    if (is_2d) {
        shape.depth = 1;
    } else {

        shape.depth = strtol((char*)num3_cstr, &end_ptr, 10);
    }

    *out = shape;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}
_parse_res _parse_u32(ts_u32* out, ts_string8 value) {
    // Checking if the string is made of valid characters
    ts_b32 is_num = true;

    for (ts_u64 i = 0; i < value.size; i++) {
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
    ts_u8* num_cstr = ts_str8_to_cstr(scratch.arena, value);
    char* end_ptr = NULL;

    ts_u32 num = strtol((char*)num_cstr, &end_ptr, 10);

    *out = num;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}
_parse_res _parse_f32(ts_f32* out, ts_string8 value) {
    // Checking if the string is made of valid characters
    ts_b32 is_num = true;

    for (ts_u64 i = 0; i < value.size; i++) {
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
    ts_u8* num_cstr = ts_str8_to_cstr(scratch.arena, value);
    char* end_ptr = NULL;

    ts_f32 num = strtof((char*)num_cstr, &end_ptr);

    *out = num;

    mga_scratch_release(scratch);

    return (_parse_res){ .error = false };
}

#define _PARSE_RES_ERR_CHECK(res) \
    if (res.error) { \
        ts_err((ts_error){ TS_ERR_PARSE, ts_str8_from_cstr((ts_u8*)res.err_msg) }); \
        goto error; \
    }

ts_b32 ts_layer_desc_load(ts_layer_desc* out, ts_string8 str) {
    if (out == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "cannot load into NULL layer desc");
        return false;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 stripped_str = ts_str8_remove_space(scratch.arena, str);

    ts_u64 colon_index = 0;
    if (!ts_str8_index_of_char(stripped_str, (ts_u8)':', &colon_index)) {
        TS_ERR(TS_ERR_PARSE, "Cannot load layer desc: Invalid string");

        goto error;
    }

    ts_string8 type_str = ts_str8_substr(stripped_str, 0, colon_index);

    out->type = ts_layer_from_name(type_str);

    ts_string8 cur_str = ts_str8_substr(stripped_str, type_str.size + 1, stripped_str.size + 1);

    while (cur_str.size > 0) {
        // Parsing of format
        // key=value;

        ts_u64 eq_index = 0;
        if (!ts_str8_index_of_char(cur_str, (ts_u8)'=', &eq_index)) {
            TS_ERR(TS_ERR_PARSE, "Cannot load layer desc: Invalid field (no '=')");
            goto error;
        }
        
        ts_u64 semi_index = 0;
        if (!ts_str8_index_of_char(cur_str, (ts_u8)';', &semi_index)) {
            TS_ERR(TS_ERR_PARSE, "Cannot load layer desc: Invalid field (no ';')\n");
            goto error;
        }

        ts_string8 key = ts_str8_substr(cur_str, 0, eq_index);
        ts_string8 value = ts_str8_substr(cur_str, eq_index + 1, semi_index);

        cur_str.str += semi_index + 1;
        cur_str.size -= semi_index + 1;

        if (key.size == 0 || value.size == 0) {
            TS_ERR(TS_ERR_PARSE, "Cannot load layer desc: Invalid key or value");
            goto error;
        }

        switch (out->type) {
            case TS_LAYER_INPUT: {
                if (ts_str8_equals(key, TS_STR8("shape"))) {
                    _parse_res res = _parse_tensor_shape(&out->input.shape, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case TS_LAYER_RESHAPE: {
                if (ts_str8_equals(key, TS_STR8("shape"))) {
                    _parse_res res = _parse_tensor_shape(&out->reshape.shape, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case TS_LAYER_DENSE: {
                if (ts_str8_equals(key, TS_STR8("size"))) {
                    _parse_res res = _parse_u32(&out->dense.size, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (ts_str8_equals(key, TS_STR8("bias_init"))) {
                    out->dense.bias_init = TS_PARAM_INIT_NULL;
                    ts_u32 bias_init = 0;

                    _parse_res res = _parse_enum(&bias_init, value, _param_init_names, TS_PARAM_INIT_COUNT);
                    out->dense.bias_init = bias_init;

                    if (res.error || out->dense.bias_init == TS_PARAM_INIT_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid param init type");
                        goto error;
                    }
                } else if (ts_str8_equals(key, TS_STR8("weight_init"))) {
                    out->dense.weight_init = TS_PARAM_INIT_NULL;
                    ts_u32 weight_init = 0;

                    _parse_res res = _parse_enum(&weight_init, value, _param_init_names, TS_PARAM_INIT_COUNT);
                    out->dense.weight_init = weight_init;

                    if (res.error || out->dense.weight_init == TS_PARAM_INIT_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid param init type");
                        goto error;
                    }
                }
            } break;
            case TS_LAYER_ACTIVATION: {
                if (ts_str8_equals(key, TS_STR8("type"))) {
                    out->activation.type = TS_ACTIVATION_NULL;
                    ts_u32 activ_type = 0;

                    _parse_res res = _parse_enum(&activ_type, value, _activ_names, TS_ACTIVATION_COUNT);
                    out->activation.type = activ_type;

                    if (res.error || out->activation.type == TS_ACTIVATION_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid activation type");
                        goto error;
                    }
                }
            } break;
            case TS_LAYER_DROPOUT: {
                if (ts_str8_equals(key, TS_STR8("keep_rate"))) {
                    _parse_res res = _parse_f32(&out->dropout.keep_rate, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case TS_LAYER_POOLING_2D: {
                if (ts_str8_equals(key, TS_STR8("type"))) {
                    out->pooling_2d.type = TS_POOLING_NULL;
                    ts_u32 pool_type = 0;

                    _parse_res res = _parse_enum(&pool_type, value, _pooling_names, TS_POOLING_COUNT);
                    out->pooling_2d.type = pool_type;

                    if (res.error || out->pooling_2d.type == TS_POOLING_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid pooling type");
                        goto error;
                    }
                } else if (ts_str8_equals(key, TS_STR8("pool_size"))) {
                    _parse_res res = _parse_tensor_shape(&out->pooling_2d.pool_size, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            case TS_LAYER_CONV_2D: {
                if (ts_str8_equals(key, TS_STR8("num_filters"))) {
                    _parse_res res = _parse_u32(&out->conv_2d.num_filters, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (ts_str8_equals(key, TS_STR8("kernel_size"))) {
                    _parse_res res = _parse_u32(&out->conv_2d.kernel_size, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (ts_str8_equals(key, TS_STR8("padding"))) {
                    _parse_res res = _parse_b32(&out->conv_2d.padding, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (ts_str8_equals(key, TS_STR8("stride"))) {
                    _parse_res res = _parse_u32(&out->conv_2d.stride, value);

                    _PARSE_RES_ERR_CHECK(res);
                } else if (ts_str8_equals(key, TS_STR8("kernels_init"))) {
                    out->conv_2d.kernels_init = TS_PARAM_INIT_NULL;
                    ts_u32 kernels_init = 0;

                    _parse_res res = _parse_enum(&kernels_init, value, _param_init_names, TS_PARAM_INIT_COUNT);
                    out->conv_2d.kernels_init = kernels_init;

                    if (res.error || out->dense.bias_init == TS_PARAM_INIT_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid param init type");
                        goto error;
                    }
                } else if (ts_str8_equals(key, TS_STR8("biases_init"))) {
                    out->conv_2d.biases_init = TS_PARAM_INIT_NULL;
                    ts_u32 biases_init = 0;

                    _parse_res res = _parse_enum(&biases_init, value, _param_init_names, TS_PARAM_INIT_COUNT);
                    out->conv_2d.biases_init = biases_init;

                    if (res.error || out->dense.bias_init == TS_PARAM_INIT_NULL) {
                        TS_ERR(TS_ERR_PARSE, "Cannot load desc: Invalid param init type");
                        goto error;
                    }
                }
            } break;
            case TS_LAYER_NORM: {
                if (ts_str8_equals(key, TS_STR8("epsilon"))) {
                    _parse_res res = _parse_f32(&out->norm.epsilon, value);

                    _PARSE_RES_ERR_CHECK(res);
                }
            } break;
            default: break;
        }
    }

    mga_scratch_release(scratch);
    return true;

error:
    mga_scratch_release(scratch);

    *out = (ts_layer_desc){ 0 };
    return false;
}

void _param_init_uniform(ts_tensor* param, ts_f32 lower, ts_f32 upper) {
    ts_u64 size = (ts_u64)param->shape.width * param->shape.height * param->shape.depth;

    mga_temp scratch = mga_scratch_get(NULL, 0);
    ts_f32* data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, size);

    for (ts_u64 i = 0; i < size; i++) {
        data[i] = ts_prng_rand_f32() * (upper - lower) + lower;
    }

    ts_tensor_set_data(param, data);

    mga_scratch_release(scratch);
}

void _param_init_normal(ts_tensor* param, ts_f32 mean, ts_f32 std_dev) {
    ts_u64 size = (ts_u64)param->shape.width * param->shape.height * param->shape.depth;

    mga_temp scratch = mga_scratch_get(NULL, 0);
    ts_f32* data = MGA_PUSH_ARRAY(scratch.arena, ts_f32, size);

    for (ts_u64 i = 0; i < size; i++) {
        data[i] = mean + ts_prng_std_norm() * std_dev;
    }

    ts_tensor_set_data(param, data);

    mga_scratch_release(scratch);
}

void ts_param_init(ts_tensor* param, ts_param_init_type input_type, ts_u64 in_size, ts_u64 out_size) {
    if (param == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot init NULL param");
        return;
    }

    switch (input_type) {
        case TS_PARAM_INIT_ZEROS: {
            ts_tensor_fill(param, 0.0f);
        } break;
        case TS_PARAM_INIT_ONES: {
            ts_tensor_fill(param, 1.0f);
        } break;
        case TS_PARAM_INIT_XAVIER_UNIFORM: {
            ts_f32 upper = sqrtf(6.0f / (ts_f32)(in_size + out_size));
            _param_init_uniform(param, -upper, upper);
        } break;
        case TS_PARAM_INIT_XAVIER_NORMAL: {
            ts_f32 std_dev = sqrtf(2.0f / (ts_f32)(in_size + out_size));
            _param_init_normal(param, 0.0f, std_dev);
        } break;
        case TS_PARAM_INIT_HE_UNIFORM: {
            ts_f32 upper = sqrtf(6.0f / (ts_f32)in_size);
            _param_init_uniform(param, -upper, upper);
        }
        case TS_PARAM_INIT_HE_NORMAL: {
            ts_f32 std_dev = sqrtf(2.0f / (ts_f32)in_size);
            _param_init_normal(param, 0.0f, std_dev);
        }
        default: break;
    }
}

void ts_layers_cache_push(ts_layers_cache* cache, ts_tensor* t) {
    if (cache == NULL || t == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot push tensor to layers cache: cache or tensor is NULL");
        return;
    }

    ts_layers_cache_node* node = MGA_PUSH_ZERO_STRUCT(cache->arena, ts_layers_cache_node);
    node->t = t;

    TS_SLL_PUSH_FRONT(cache->first, cache->last, node);
}
ts_tensor* ts_layers_cache_pop(ts_layers_cache* cache) {
    if (cache == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot pop tensor from NULL cache");
        return NULL;
    }

    ts_tensor* out = cache->first->t;

    TS_SLL_POP_FRONT(cache->first, cache->last);

    return out;
}
