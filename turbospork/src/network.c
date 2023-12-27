#include "network.h"
#include "layers/layers_internal.h"
#include "err.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ts_u32 _network_max_layer_size(const ts_network* nn) {
    ts_u64 max_layer_size = 0;
    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_tensor_shape s = nn->layers[i]->shape;

        ts_u64 size = (ts_u64)s.width * s.height * s.depth;

        if (size > max_layer_size) {
            max_layer_size = size;
        }
    }

    return max_layer_size;
}

// Checks that each layer outputs the correct shape
// Called after input layer check and max_layer_size
ts_b32 _network_shape_checks(const ts_network* nn) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    // Performing a mock feedforward and backprop to check shapes
    ts_tensor* in_out = ts_tensor_create_alloc(scratch.arena, nn->layers[0]->shape, nn->max_layer_size);
    ts_layers_cache cache = { .arena = scratch.arena };

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_feedforward(nn->layers[i], in_out, &cache);

        if (!ts_tensor_shape_eq(in_out->shape, nn->layers[i]->shape)) {
            goto fail;
        }
    }

    // Renaming for clarity
    ts_tensor* delta = in_out;

    for (ts_i64 i = nn->num_layers - 1; i >= 0; i--) {
        ts_layer_backprop(nn->layers[i], delta, &cache);

        if (!ts_tensor_shape_eq(delta->shape, nn->layers[TS_MAX(0, i-1)]->shape)) {
            goto fail;
        }
    }

    mga_scratch_release(scratch);
    return true;

fail:
    mga_scratch_release(scratch);
    return false;
}

ts_network* ts_network_create(mg_arena* arena, ts_u32 num_layers, const ts_layer_desc* layer_descs, ts_b32 training_mode) {
    if (layer_descs == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create network with NULL layer_descs");

        return NULL;
    }

    mga_temp maybe_temp = mga_temp_begin(arena);
    ts_network* nn = MGA_PUSH_ZERO_STRUCT(arena, ts_network);

    nn->training_mode = training_mode;
    nn->num_layers = num_layers;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, ts_layer_desc, nn->num_layers);
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, ts_layer*, nn->num_layers);

    ts_tensor_shape prev_shape = { 0 };
    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        nn->layer_descs[i] = ts_layer_desc_apply_default(&layer_descs[i]);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = ts_layer_create(arena, &nn->layer_descs[i], prev_shape);

        if (nn->layers[i] == NULL) {
            TS_ERR(TS_ERR_CREATE, "Cannot create network: failed to create layer");

            goto error;
        }

        prev_shape = nn->layers[i]->shape;
    }

    if (nn->layers[0]->type != TS_LAYER_INPUT) {
        TS_ERR(TS_ERR_INVALID_INPUT, "First layer of network must be input");
        goto error;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    if (!_network_shape_checks(nn)) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create network: layer shapes do not align");
        goto error;
    }

    return nn;

error:
    mga_temp_end(maybe_temp);
    return NULL;
}

// Inits layers from stripped tsl string
// See ts_network_save_layout for more detail
static ts_b32 _ts_network_load_layout_impl(mg_arena* arena, ts_network* nn, ts_string8 file, ts_b32 training_mode) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    // Each string in list is a layer_desc save str
    ts_string8_list desc_str_list = { 0 };

    ts_u64 desc_str_start = 0;
    ts_u64 last_semi = 0;
    ts_b32 first_colon = true;
    for (ts_u64 i = 0; i < file.size; i++) {
        ts_u8 c = file.str[i];

        if (c == ';') {
            last_semi = i;

            continue;
        }

        // Colon triggers start of new desc
        // So the old one gets pushed onto the list
        if (c == ':') {
            // If it is the first colon, the string should not be saved 
            if (first_colon) {
                first_colon = false;

                continue;
            }

            ts_string8 desc_str = ts_str8_substr(file, desc_str_start, last_semi + 1);

            ts_str8_list_push(scratch.arena, &desc_str_list, desc_str);

            desc_str_start = last_semi + 1;

            // This makes it so that layers without parameters still work correctly
            // (Layers without params would have no semi colons)
            last_semi = i;
        }
    }
    ts_string8 last_str = ts_str8_substr(file, desc_str_start, file.size);
    ts_str8_list_push(scratch.arena, &desc_str_list, last_str);

    nn->num_layers = desc_str_list.node_count;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, ts_layer_desc, nn->num_layers);
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, ts_layer*, nn->num_layers);

    ts_string8_node* n = desc_str_list.first;
    ts_tensor_shape prev_shape = { 0 };
    for (ts_u32 i = 0; i < nn->num_layers; i++, n = n->next) {
        if (!ts_layer_desc_load(&nn->layer_descs[i], n->str)) {
            goto error;
        }

        nn->layer_descs[i] = ts_layer_desc_apply_default(&nn->layer_descs[i]);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = ts_layer_create(arena, &nn->layer_descs[i], prev_shape);

        if (nn->layers[i] == NULL) {
            goto error;
        }

        prev_shape = nn->layers[i]->shape;
    }

    if (nn->layers[0]->type != TS_LAYER_INPUT) {
        TS_ERR(TS_ERR_INVALID_INPUT, "First layer of network must be input");

        goto error;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    mga_scratch_release(scratch);

    if (!_network_shape_checks(nn)) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot create network: layer shapes do not align");
        goto error;
    }

    return true;

error:
    mga_scratch_release(scratch);
    return false;
}

// Creates ts_network from layout file (*.tsl)
ts_network* ts_network_load_layout(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode) {
    mga_temp maybe_temp = mga_temp_begin(arena);
    ts_network* nn = MGA_PUSH_ZERO_STRUCT(arena, ts_network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 raw_file = ts_file_read(scratch.arena, file_name);
    ts_string8 file = ts_str8_remove_space(scratch.arena, raw_file);

    if (!_ts_network_load_layout_impl(arena, nn, file, training_mode)) {
        mga_temp_end(maybe_temp);
        mga_scratch_release(scratch);

        return NULL;
    }

    mga_scratch_release(scratch);

    return nn;
}

// This is also used in ts_network_save
static const ts_string8 _tsn_header = {
    .size = 10,
    .str = (ts_u8*)"TS_network"
};

// Creates ts_network from ts_network file (*.tsn)
ts_network* ts_network_load(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode) {
    mga_temp maybe_temp = mga_temp_begin(arena);
    ts_network* nn = MGA_PUSH_ZERO_STRUCT(arena, ts_network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 file = ts_file_read(scratch.arena, file_name);

    if (!ts_str8_equals(_tsn_header, ts_str8_substr(file, 0, _tsn_header.size))) {
        TS_ERR(TS_ERR_PARSE, "Cannot load ts_network: not tsn file");

        goto error;
    }

    file = ts_str8_substr(file, _tsn_header.size, file.size);

    ts_u64 tst_index = 0;
    if (!ts_str8_index_of(file, ts_tensor_get_tst_header(), &tst_index)) {
        TS_ERR(TS_ERR_PARSE, "Cannot load ts_network: invalid tsn file");

        goto error;
    }

    ts_string8 layout_str = ts_str8_substr(file, 0, tst_index);
    ts_string8 ts_tensors_str = ts_str8_substr(file, tst_index, file.size);

    if (!_ts_network_load_layout_impl(arena, nn, layout_str, training_mode)) {
        goto error;
    }

    ts_tensor_list params = ts_tensor_list_from_str(scratch.arena, ts_tensors_str);

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_load(nn->layers[i], &params, i);
    }

    mga_scratch_release(scratch);
    return nn;

error:
    mga_temp_end(maybe_temp);
    mga_scratch_release(scratch);

    return NULL;
}

void ts_network_delete(ts_network* nn) {
    if (nn == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot delete NULL network");
        return;
    }

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_delete(nn->layers[i]);
    }
}

void ts_network_feedforward(const ts_network* nn, ts_tensor* out, const ts_tensor* input) {
    if (nn == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot feedforward NULL network");
        return;
    }
    if (out == NULL || input == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot feedforward with NULL input and/or output");
        return;
    }

    ts_u64 input_size = (ts_u64)input->shape.width * input->shape.height * input->shape.depth;
    ts_tensor_shape nn_shape = nn->layers[0]->shape;
    ts_u64 nn_input_size = (ts_u64)nn_shape.width * nn_shape.height * nn_shape.depth;

    if (input_size != nn_input_size) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Input must be as big as the network input layer");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_tensor* in_out = ts_tensor_create_alloc(scratch.arena, (ts_tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    ts_tensor_copy_ip(in_out, input);

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_feedforward(nn->layers[i], in_out, NULL);
    }

    ts_tensor_copy_ip(out, in_out);

    mga_scratch_release(scratch);
}

ts_u32 _num_digits (ts_u32 n) {
    if (n < 10) return 1;
    if (n < 100) return 2;
    if (n < 1000) return 3;
    if (n < 10000) return 4;
    if (n < 100000) return 5;
    if (n < 1000000) return 6;
    if (n < 10000000) return 7;
    if (n < 100000000) return 8;
    if (n < 1000000000) return 9;
    return 10;
}

typedef struct {
    ts_network* nn;
    ts_tensor input_view;
    ts_tensor output_view;
    ts_cost_type cost;
} _ts_network_backprop_args;

void _ts_network_backprop_thread(void* args) {
    _ts_network_backprop_args* bargs = (_ts_network_backprop_args*)args;

    ts_network* nn = bargs->nn;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_layers_cache cache = { .arena = scratch.arena };

    ts_tensor* in_out = ts_tensor_create_alloc(scratch.arena, (ts_tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    ts_tensor_copy_ip(in_out, &bargs->input_view);
    ts_tensor* output = ts_tensor_copy(scratch.arena, &bargs->output_view, false);

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_feedforward(nn->layers[i], in_out, &cache);
    }

    // Renaming for clarity
    ts_tensor* delta = in_out;
    ts_cost_grad(bargs->cost, delta, output);

    for (ts_i64 i = nn->num_layers - 1; i >= 0; i--) {
        ts_layer_backprop(nn->layers[i], delta, &cache);
    }

    mga_scratch_release(scratch);
}

typedef struct {
    ts_u32* num_correct;
    ts_mutex* num_correct_mutex;

    ts_network* nn;

    ts_tensor input_view;
    ts_tensor_index output_argmax;
} _ts_network_test_args;
void _ts_network_test_thread(void* args) {
    _ts_network_test_args* targs = (_ts_network_test_args*)args;

    ts_network* nn = targs->nn;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_tensor* in_out = ts_tensor_create_alloc(scratch.arena, (ts_tensor_shape){ 1, 1, 1 }, nn->max_layer_size);
    ts_tensor_copy_ip(in_out, &targs->input_view);

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_feedforward(nn->layers[i], in_out, NULL);
    }

    if (ts_tensor_index_eq(ts_tensor_argmax(in_out), targs->output_argmax)) {
        ts_mutex_lock(targs->num_correct_mutex);

        *targs->num_correct += 1;

        ts_mutex_unlock(targs->num_correct_mutex);
    }
    
    mga_scratch_release(scratch);
}

#define _BAR_SIZE 20
void ts_network_train(ts_network* nn, const ts_network_train_desc* desc) {
    if (nn == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot train NULL network");
        return;
    }
    if (!nn->training_mode) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot train network that is not in training mode");
        return;
    }

    // Size checks
    {
        ts_tensor_shape nn_shape = nn->layers[0]->shape;
        ts_u64 nn_input_size = (ts_u64)nn_shape.width * nn_shape.height * nn_shape.depth;
        nn_shape = nn->layers[nn->num_layers - 1]->shape;
        ts_u64 nn_out_size = (ts_u64)nn_shape.width * nn_shape.height * nn_shape.depth;

        ts_u64 input_size = (ts_u64)desc->train_inputs->shape.width * desc->train_inputs->shape.height;
        if (input_size != nn_input_size) {
            TS_ERR(TS_ERR_INVALID_INPUT, "Training inputs must be the same size as the network input layer");
            return;
        }
        ts_u64 out_size = (ts_u64)desc->train_outputs->shape.width * desc->train_outputs->shape.height;
        if (out_size != nn_out_size) {
            TS_ERR(TS_ERR_INVALID_INPUT, "Training outpus must be the same size as the network output layer");
            return;
        }

        if (desc->accuracy_test) {
            input_size = (ts_u64)desc->test_inputs->shape.width * desc->test_inputs->shape.height;
            if (input_size != nn_input_size) {
                TS_ERR(TS_ERR_INVALID_INPUT, "Testing inputs must be the same size as the network input layer");
                return;
            }
            out_size = (ts_u64)desc->test_outputs->shape.width * desc->test_outputs->shape.height;
            if (out_size != nn_out_size) {
                TS_ERR(TS_ERR_INVALID_INPUT, "Testing outpus must be the same size as the network output layer");
                return;
            }
        }
    }

    ts_optimizer optim = desc->optim;
    optim._batch_size = desc->batch_size;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // +1 is just for insurance
    ts_thread_pool* tpool = ts_thread_pool_create(scratch.arena, TS_MAX(1, desc->num_threads), desc->batch_size + 1);

    _ts_network_backprop_args* backprop_args = MGA_PUSH_ZERO_ARRAY(scratch.arena, _ts_network_backprop_args, desc->batch_size);

    // Accuracy testing stuff
    _ts_network_test_args* test_args = NULL;
    ts_u32 num_correct = 0;
    ts_mutex* num_correct_mutex = NULL;
    if (desc->accuracy_test) {
        test_args = MGA_PUSH_ZERO_ARRAY(scratch.arena, _ts_network_test_args, desc->batch_size);
        num_correct_mutex = ts_mutex_create(scratch.arena);
    }

    ts_u8 bar_str_data[_BAR_SIZE + 1] = { 0 };
    memset(bar_str_data, ' ', _BAR_SIZE);

    ts_u8 batch_str_data[11] = { 0 };

    // This will add one if there is a remainder
    // Allows for batch sizes that are not perfectly divisible
    div_t num_batches_div = div(desc->train_inputs->shape.depth, desc->batch_size);
    ts_u32 num_batches = num_batches_div.quot + (num_batches_div.rem != 0);
    ts_u32 last_batch_size = desc->train_inputs->shape.depth - (desc->batch_size * (num_batches - 1));

    ts_u32 num_batches_digits = _num_digits(num_batches);

    // Same calculations for test batches
    div_t num_test_batches_div = div(desc->test_inputs->shape.depth, desc->batch_size);
    ts_u32 num_test_batches = num_test_batches_div.quot + (num_test_batches_div.rem != 0);
    ts_u32 last_test_batch_size = desc->test_inputs->shape.depth - (desc->batch_size * (num_test_batches - 1));

    for (ts_u32 epoch = 0; epoch < desc->epochs; epoch++) {
        printf("Epoch: %u / %u\n", epoch + 1, desc->epochs);

        for (ts_u32 batch = 0; batch < num_batches; batch++) {
            // Progress in stdout
            {
                // This is so the batch number always takes up the same amount of space
                ts_u32 batch_digits = _num_digits(batch + 1);
                memset(batch_str_data, ' ', 9);
                ts_u32 offset = num_batches_digits - batch_digits;
                snprintf((char*)(batch_str_data + offset), 11 - offset, "%u", batch + 1);
                printf("%.*s / %u  ", (int)num_batches_digits, batch_str_data, num_batches);

                ts_f32 bar_length = (ts_f32)_BAR_SIZE * ((ts_f32)(batch + 1) / num_batches);
                ts_u32 bar_chars = ceilf(bar_length);
                memset(bar_str_data, '=', bar_chars);
                if (batch + 1 != num_batches) {
                    bar_str_data[bar_chars - 1] = '>';
                }

                printf("[%s]", bar_str_data);

                printf("\r");

                fflush(stdout);
            }

            mga_temp batch_temp = mga_temp_begin(scratch.arena);

            // Training batch
            ts_u32 batch_size = (batch == num_batches - 1) ? last_batch_size : desc->batch_size;
            for (ts_u32 i = 0; i < batch_size; i++) {
                ts_u64 index = (ts_u64)i + (ts_u64)batch * desc->batch_size;

                ts_tensor input_view = { 0 };
                ts_tensor output_view = { 0 };
                ts_tensor_2d_view(&input_view, desc->train_inputs, index);
                ts_tensor_2d_view(&output_view, desc->train_outputs, index);

                backprop_args[i] = (_ts_network_backprop_args){ 
                    .nn = nn,
                    .input_view = input_view,
                    .output_view = output_view,
                    .cost = desc->cost,
                };

                ts_thread_pool_add_task(
                    tpool,
                    (ts_thread_task){
                        .func = _ts_network_backprop_thread,
                        .arg = &backprop_args[i]
                    }
                );
            }

            ts_thread_pool_wait(tpool);

            for (ts_u32 i = 0; i < nn->num_layers; i++) {
                ts_layer_apply_changes(nn->layers[i], &optim);
            }

            mga_temp_end(batch_temp);
        }

        printf("\n");
        memset(bar_str_data, ' ', _BAR_SIZE);

        if (desc->save_interval != 0 && ((epoch + 1) % desc->save_interval) == 0) {
            mga_temp save_temp = mga_temp_begin(scratch.arena);

            ts_string8 path = ts_str8_pushf(save_temp.arena, "%.*s%.4u.tsn", (int)desc->save_path.size, desc->save_path.str, epoch + 1);

            ts_network_save(nn, path);

            mga_temp_end(save_temp);
        }

        ts_f32 accuracy = 0.0f;
        if (desc->accuracy_test) {
            num_correct = 0;

            ts_time_init();
            ts_string8 load_anim = TS_STR8("-\\|/");
            ts_u64 anim_start_time = ts_now_usec();
            ts_u32 anim_frame = 0;

            // Accuracy test is also done in batches for multithreading
            for (ts_u32 batch = 0; batch < num_test_batches; batch++) {
                ts_u64 cur_time = ts_now_usec();
                if (cur_time - anim_start_time > 100000) {
                    anim_start_time = cur_time;
                    anim_frame++;

                    printf("Test Accuracy: %c\r", load_anim.str[anim_frame % load_anim.size]);
                    fflush(stdout);
                }

                mga_temp batch_temp = mga_temp_begin(scratch.arena);

                // Test batch
                ts_u32 batch_size = batch == num_test_batches - 1 ? last_test_batch_size : desc->batch_size;
                for (ts_u32 i = 0; i < batch_size; i++) {
                    ts_u64 index = (ts_u64)i + (ts_u64)batch * desc->batch_size;

                    ts_tensor input_view = { 0 };
                    ts_tensor output_view = { 0 };
                    ts_tensor_2d_view(&input_view, desc->test_inputs, index);
                    ts_tensor_2d_view(&output_view, desc->test_outputs, index);

                    ts_tensor_index output_argmax = ts_tensor_argmax(&output_view);

                    test_args[i] = (_ts_network_test_args){ 
                        .num_correct = &num_correct,
                        .num_correct_mutex = num_correct_mutex,

                        .nn = nn,
                        .input_view = input_view,
                        .output_argmax = output_argmax
                    };

                    ts_thread_pool_add_task(
                        tpool,
                        (ts_thread_task){
                            .func = _ts_network_test_thread,
                            .arg = &test_args[i]
                        }
                    );
                }

                ts_thread_pool_wait(tpool);

                mga_temp_end(batch_temp);
            }

            accuracy = (ts_f32)num_correct / desc->test_inputs->shape.depth;

            printf("Test Accuracy: %f\n", accuracy);
        }

        if (desc->epoch_callback) {
            ts_network_epoch_info info = {
                .epoch = epoch,

                .test_accuracy = accuracy
            };

            desc->epoch_callback(&info);
        }
    }

    ts_thread_pool_destroy(tpool);

    if (desc->accuracy_test) {
        ts_mutex_destroy(num_correct_mutex);
    }

    mga_scratch_release(scratch);
}

/*
Sample Summary:

-------------------------
  ts_network (5 layers)

type        shape
----        -----
input       (784, 1, 1)
dense       (64, 1, 1)
activation  (64, 1, 1)
dense       (10, 1, 1)
activation  (10, 1, 1)

-------------------------
*/
void ts_network_summary(const ts_network* nn) {
    if (nn == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot print summary of NULL network");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8 header = ts_str8_pushf(scratch.arena, "network (%u layers)", nn->num_layers);

    // Storing strings in a list first to get good spacing in the console
    // +2 is for column name and "---" separator
    ts_string8* types = MGA_PUSH_ZERO_ARRAY(scratch.arena, ts_string8, nn->num_layers + 2);
    ts_string8* shapes = MGA_PUSH_ZERO_ARRAY(scratch.arena, ts_string8, nn->num_layers + 2);

    types[0] = TS_STR8("type");
    types[1] = TS_STR8("----");

    shapes[0] = TS_STR8("shape");
    shapes[1] = TS_STR8("-----");

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        types[i + 2] = ts_layer_get_name(nn->layers[i]->type);

        ts_tensor_shape s = nn->layers[i]->shape;
        ts_string8 shape_str = ts_str8_pushf(scratch.arena, "(%u %u %u)", s.width, s.height, s.depth);

        shapes[i + 2] = shape_str;
    }

    ts_u64 max_type_width = types[0].size;
    ts_u64 max_shape_width = shapes[0].size;

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        if (types[i + 2].size > max_type_width) {
            max_type_width = types[i + 2].size;
        }

        if (shapes[i + 2].size > max_shape_width) {
            max_shape_width = shapes[i + 2].size;
        }
    }

    // Spacing added before, between, and after items
    ts_u64 row_width = 1 + max_type_width + 2 + max_shape_width + 1;
    row_width = TS_MAX(row_width, header.size + 2);

    // For even spacing of the header
    if ((row_width - header.size) % 2 != 0) {
        row_width += 1;
    }
    
    // For newline
    row_width++;

    // Borders + border padding + header + layers + titles
    ts_u32 num_rows = 2 + 2 + 1 + nn->num_layers + 2;

    ts_string8 out = {
        .size = row_width * num_rows,
        .str = MGA_PUSH_ARRAY(scratch.arena, ts_u8, row_width * num_rows)
    };

    memset(out.str, ' ', out.size);
    for (ts_u32 y = 0; y < num_rows; y++) {
        out.str[row_width - 1 + y * row_width] = '\n';
    }

    // Borders
    memset(out.str, '-', row_width - 1);
    memset(out.str + (num_rows - 1) * row_width, '-', row_width - 1);

    // Header
    ts_u32 header_spacing = (row_width - 1 - header.size) / 2;
    memcpy(out.str + row_width + header_spacing, header.str, header.size);

    ts_u32 shape_start_x = 1 + max_type_width + 2;
    for (ts_u32 i = 0; i < nn->num_layers + 2; i++) {
        // Start index into row
        ts_u64 start_i = (i + 3) * row_width;

        memcpy(out.str + start_i + 1, types[i].str, types[i].size);
        memcpy(out.str + start_i + shape_start_x, shapes[i].str, shapes[i].size);
    }

    printf("%.*s", (int)out.size, (char*)out.str);

    mga_scratch_release(scratch);
}

/*
File Format (*.tsl):

List of layer_desc saves
See layer_desc_save
*/
void ts_network_save_layout(const ts_network* nn, ts_string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_string8_list save_list = { 0 };

    // For spacing between layer_descs
    ts_string8 new_line = TS_STR8("\n");

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_desc_save(scratch.arena, &save_list, &nn->layer_descs[i]);
        ts_str8_list_push(scratch.arena, &save_list, new_line);
    }

    ts_file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

/*
File Format (*.tsn):

Header
ts_network Layout (tsl)
ts_tensor List of layer params
*/
void ts_network_save(const ts_network* nn, ts_string8 file_name) {
    if (nn == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Cannot save NULL network");
        return;
    }

    mga_temp scratch = mga_scratch_get(NULL, 0);
    ts_string8 layout_str = { 0 };

    {
        mga_temp scratch2 = mga_scratch_get(&scratch.arena, 1);

        ts_string8_list layout_list = { 0 };
        for (ts_u32 i = 0; i < nn->num_layers; i++) {
            ts_layer_desc_save(scratch.arena, &layout_list, &nn->layer_descs[i]);
        }

        ts_string8 full_layout_str = ts_str8_concat(scratch2.arena, layout_list);
        layout_str = ts_str8_remove_space(scratch.arena, full_layout_str);

        mga_scratch_release(scratch2);
    }


    ts_tensor_list param_list = { 0 };
    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_save(scratch.arena, nn->layers[i], &param_list, i);
    }

    ts_string8 param_str = ts_tensor_list_to_str(scratch.arena, &param_list);

    ts_string8_list save_list = { 0 };
    ts_str8_list_push(scratch.arena, &save_list, _tsn_header);
    ts_str8_list_push(scratch.arena, &save_list, layout_str);
    ts_str8_list_push(scratch.arena, &save_list, param_str);

    ts_file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

