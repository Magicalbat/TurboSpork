#include "network.h"
#include "layers/layers_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ts_u32 _network_max_layer_size(ts_network* nn) {
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

ts_network* ts_network_create(mg_arena* arena, ts_u32 num_layers, const ts_layer_desc* layer_descs, ts_b32 training_mode) {
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
        prev_shape = nn->layers[i]->shape;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    return nn;
}

// Inits layers from stripped tpl string
// See ts_network_save_layout for more detail
static void _ts_network_load_layout_impl(mg_arena* arena, ts_network* nn, ts_string8 file, ts_b32 training_mode) {
    // TODO: Error checking for missing semicolons

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
        ts_layer_desc desc = ts_layer_desc_load(n->str);
        nn->layer_descs[i] = ts_layer_desc_apply_default(&desc);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = ts_layer_create(arena, &nn->layer_descs[i], prev_shape);
        prev_shape = nn->layers[i]->shape;
    }

    nn->max_layer_size = _network_max_layer_size(nn); 

    mga_scratch_release(scratch);
}

// Creates ts_network from layout file (*.tpl)
ts_network* ts_network_load_layout(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode) {
    ts_network* nn = MGA_PUSH_ZERO_STRUCT(arena, ts_network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 raw_file = ts_file_read(scratch.arena, file_name);
    ts_string8 file = ts_str8_remove_space(scratch.arena, raw_file);

    _ts_network_load_layout_impl(arena, nn, file, training_mode);

    mga_scratch_release(scratch);

    return nn;
}

// This is also used in ts_network_save
static const ts_string8 _tpn_header = {
    .size = 10,
    .str = (ts_u8*)"TP_ts_network"
};

// Creates ts_network from ts_network file (*.tpn)
ts_network* ts_network_load(mg_arena* arena, ts_string8 file_name, ts_b32 training_mode) {
    ts_network* nn = MGA_PUSH_ZERO_STRUCT(arena, ts_network);

    nn->training_mode = training_mode;

    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 file = ts_file_read(scratch.arena, file_name);

    if (!ts_str8_equals(_tpn_header, ts_str8_substr(file, 0, _tpn_header.size))) {
        fprintf(stderr, "Cannot load ts_network: not tpn file\n");

        goto end;
    }

    file = ts_str8_substr(file, _tpn_header.size, file.size);

    ts_u64 tpt_index = 0;
    if (!ts_str8_index_of(file, ts_tensor_get_tpt_header(), &tpt_index)) {
        fprintf(stderr, "Cannot load ts_network: invalid tpn file\n");

        goto end;
    }

    ts_string8 layout_str = ts_str8_substr(file, 0, tpt_index);
    ts_string8 ts_tensors_str = ts_str8_substr(file, tpt_index, file.size);

    _ts_network_load_layout_impl(arena, nn, layout_str, training_mode);

    ts_tensor_list params = ts_tensor_list_from_str(scratch.arena, ts_tensors_str);

    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_load(nn->layers[i], &params, i);
    }

    // Using goto so that scratch arena always gets released
end:
    mga_scratch_release(scratch);
    return nn;
}

void ts_network_delete(ts_network* nn) {
    for (ts_u32 i = 0; i < nn->num_layers; i++) {
        ts_layer_delete(nn->layers[i]);
    }
}

void ts_network_feedforward(const ts_network* nn, ts_tensor* out, const ts_tensor* input) {
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
    /*      2147483647 is 2^31-1 - add more ifs as needed
       and adjust this final return as well. */
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
    if (!nn->training_mode) {
        fprintf(stderr, "Cannot train ts_network that is not in training mode\n");

        return;
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

            ts_string8 path = ts_str8_pushf(save_temp.arena, "%.*s%.4u.tpn", (int)desc->save_path.size, desc->save_path.str, epoch + 1);

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

    // Spacing before, between, and after items
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
File Format (*.tpl):

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

ts_string8 ts_network_get_tpn_header(void) {
    return _tpn_header;
}

/*
File Format (*.tpn):

Header
ts_network Layout (tpl)
ts_tensor List of layer params
*/
void ts_network_save(const ts_network* nn, ts_string8 file_name) {
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
        ts_layer_save(scratch.arena, &param_list, nn->layers[i], i);
    }

    ts_string8 param_str = ts_tensor_list_to_str(scratch.arena, &param_list);

    ts_string8_list save_list = { 0 };
    ts_str8_list_push(scratch.arena, &save_list, _tpn_header);
    ts_str8_list_push(scratch.arena, &save_list, layout_str);
    ts_str8_list_push(scratch.arena, &save_list, param_str);

    ts_file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

