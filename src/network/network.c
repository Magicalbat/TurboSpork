#include "network.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static u64 _network_max_layer_size(const network* nn);

network* network_create(mg_arena* arena, u32 num_layers, const layer_desc* layer_descs, b32 training_mode) {
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    nn->num_layers = num_layers;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, layer_desc, nn->num_layers);
    memcpy(nn->layer_descs, layer_descs, sizeof(layer_desc) * nn->num_layers);

    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, layer*, nn->num_layers);

    tensor_shape prev_shape = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++) {
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = layer_create(arena, &nn->layer_descs[i], prev_shape);
        prev_shape = nn->layers[i]->shape;
    }

    return nn;
}

// Inits layers from stripped tpl string
// See network_save_layout for more detail
static void _network_load_layout_impl(mg_arena* arena, network* nn, string8 file, b32 training_mode) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    // Each string in list is a layer_desc save str
    string8_list desc_str_list = { 0 };

    u64 desc_str_start = 0;
    u64 last_semi = 0;
    b32 first_colon = true;
    for (u64 i = 0; i < file.size; i++) {
        u8 c = file.str[i];

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

            string8 desc_str = str8_substr(file, desc_str_start, last_semi + 1);

            str8_list_push(scratch.arena, &desc_str_list, desc_str);

            desc_str_start = last_semi + 1;

            // This makes it so that layers without parameters still work correctly
            // (Layers without params would have no semi colons)
            last_semi = i;
        }
    }
    string8 last_str = str8_substr(file, desc_str_start, file.size);
    str8_list_push(scratch.arena, &desc_str_list, last_str);

    nn->num_layers = desc_str_list.node_count;

    nn->layer_descs = MGA_PUSH_ZERO_ARRAY(arena, layer_desc, nn->num_layers);
    nn->layers = MGA_PUSH_ZERO_ARRAY(arena, layer*, nn->num_layers);

    string8_node* n = desc_str_list.first;
    tensor_shape prev_shape = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++, n = n->next) {
        nn->layer_descs[i] = layer_desc_load(n->str);
        nn->layer_descs[i].training_mode = training_mode;

        nn->layers[i] = layer_create(arena, &nn->layer_descs[i], prev_shape);
        prev_shape = nn->layers[i]->shape;
    }

    mga_scratch_release(scratch);
}

// Creates network from layout file (*.tpl)
network* network_load_layout(mg_arena* arena, string8 file_name, b32 training_mode) {
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 raw_file = os_file_read(scratch.arena, file_name);
    string8 file = str8_remove_space(scratch.arena, raw_file);

    _network_load_layout_impl(arena, nn, file, training_mode);

    mga_scratch_release(scratch);

    return nn;
}

// This is also used in network_save
static const string8 _tpn_header = {
    .size = 10,
    .str = (u8*)"TP_network"
};

// Creates network from network file (*.tpn)
network* network_load(mg_arena* arena, string8 file_name, b32 training_mode) {
    network* nn = MGA_PUSH_ZERO_STRUCT(arena, network);

    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 file = os_file_read(scratch.arena, file_name);

    if (!str8_equals(_tpn_header, str8_substr(file, 0, _tpn_header.size))) {
        fprintf(stderr, "Cannot load network: not tpn file\n");

        goto end;
    }

    file = str8_substr(file, _tpn_header.size, file.size);

    u64 tpt_index = 0;
    if (!str8_index_of(file, tensor_get_tpt_header(), &tpt_index)) {
        fprintf(stderr, "Cannot load network: invalid tpn file\n");

        goto end;
    }

    string8 layout_str = str8_substr(file, 0, tpt_index);
    string8 tensors_str = str8_substr(file, tpt_index, file.size);

    _network_load_layout_impl(arena, nn, layout_str, training_mode);

    tensor_list params = tensor_list_from_str(scratch.arena, tensors_str);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_load(nn->layers[i], &params, i);
    }

    // Using goto so that scratch arena always gets released
end:
    mga_scratch_release(scratch);
    return nn;
}

void network_delete(network* nn) {
    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_delete(nn->layers[i]);
    }
}

void network_feedforward(const network* nn, tensor* out, const tensor* input) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    u64 max_layer_size = _network_max_layer_size(nn);

    tensor* in_out = tensor_create_alloc(scratch.arena, (tensor_shape){ 1, 1, 1 }, max_layer_size);
    tensor_copy_ip(in_out, input);

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, NULL);
    }

    tensor_copy_ip(out, in_out);

    mga_scratch_release(scratch);
}

u32 _num_digits (u32 n) {
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
    network* nn;
    tensor* in_out;
    tensor* output;
    cost_type cost;
} _network_backprop_args;
void _network_backprop_thread(void* args) {
    _network_backprop_args* bargs = (_network_backprop_args*)args;

    network* nn = bargs->nn;
    tensor* in_out = bargs->in_out;

    mga_temp scratch = mga_scratch_get(NULL, 0);
    layers_cache cache = { .arena = scratch.arena };

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_feedforward(nn->layers[i], in_out, &cache);
    }

    // Renaming for clarity
    tensor* delta = in_out;
    cost_grad(bargs->cost, delta, bargs->output);

    for (i64 i = nn->num_layers - 1; i >= 0; i--) {
        layer_backprop(nn->layers[i], delta, &cache);
    }

    mga_scratch_release(scratch);
}

#define _BAR_SIZE 20
void network_train(network* nn, const network_train_desc* desc) {
    optimizer optim = desc->optim;
    optim._batch_size = desc->batch_size;

    mga_temp scratch = mga_scratch_get(NULL, 0);

    // +1 is just for insurance
    os_thread_pool* tpool = os_thread_pool_create(scratch.arena, MAX(1, desc->num_threads), desc->batch_size + 1);
    _network_backprop_args* backprop_args = MGA_PUSH_ZERO_ARRAY(scratch.arena, _network_backprop_args, desc->batch_size);

    u8 bar_str_data[_BAR_SIZE + 1] = { 0 };
    memset(bar_str_data, ' ', _BAR_SIZE);

    u8 batch_str_data[10] = { 0 };

    // TODO: detect windows cmd?

    // Hides cursor
    //printf("\e[?25l");

    for (u32 epoch = 0; epoch < desc->epochs; epoch++) {
        printf("Epoch: %u / %u\n", epoch + 1, desc->epochs);

        u32 num_batches = desc->train_inputs->shape.depth / desc->batch_size;
        u32 num_batches_digits = _num_digits(num_batches);

        for (u32 batch = 0; batch < num_batches; batch++) {
            // Progress in stdout
            {
                // This is so the batch number always takes up the same amount of space
                u32 batch_digits = _num_digits(batch + 1);
                memset(batch_str_data, ' ', 9);
                snprintf((char*)(batch_str_data + (num_batches_digits - batch_digits)), 10, "%u", batch + 1);
                printf("%.*s / %u  ", (int)num_batches_digits, batch_str_data, num_batches);

                f32 bar_length = (f32)_BAR_SIZE * ((f32)(batch + 1) / num_batches);
                u32 bar_chars = ceilf(bar_length);
                memset(bar_str_data, '=', bar_chars);
                if (batch + 1 != num_batches) {
                    bar_str_data[bar_chars - 1] = '>';
                }

                printf("[%s]", bar_str_data);

                printf("\r");
            }

            mga_temp batch_temp = mga_temp_begin(scratch.arena);

            // Training batch
            for (u32 i = 0; i < desc->batch_size; i++) {
                u64 index = (u64)i + (u64)batch * desc->batch_size;

                tensor input_view = { 0 };
                tensor output_view = { 0 };
                tensor_2d_view(&input_view, desc->train_inputs, index);
                tensor_2d_view(&output_view, desc->train_outputs, index);

                u64 max_layer_size = _network_max_layer_size(nn);

                tensor* in_out = tensor_create_alloc(batch_temp.arena, (tensor_shape){ 1, 1, 1 }, max_layer_size);
                tensor_copy_ip(in_out, &input_view);
                tensor* output = tensor_copy(batch_temp.arena, &output_view, false);

                backprop_args[i] = (_network_backprop_args){ 
                    .nn = nn,
                    .in_out = in_out,
                    .output = output,
                    .cost = desc->cost,
                };

                // Stopping here?
                os_thread_pool_add_task(
                    tpool,
                    (os_thread_task){
                        .func = _network_backprop_thread,
                        .arg = &backprop_args[i]
                    }
                );
            }

            os_thread_pool_wait(tpool);

            for (u32 i = 0; i < nn->num_layers; i++) {
                layer_apply_changes(nn->layers[i], &optim);
            }

            mga_temp_end(batch_temp);
        }

        printf("\n");
        memset(bar_str_data, ' ', _BAR_SIZE);

        f32 accuracy = -1.0f;

        if (desc->accuracy_test) {
            string8 load_anim = STR8("-\\|/");

            u32 num_correct = 0;
            tensor* out = tensor_create(scratch.arena, (tensor_shape){ 10, 1, 1 });
            tensor view = { 0 };

            for (u32 i = 0; i < desc->test_inputs->shape.depth; i++) {
                printf("Test Accuracy: %c\r", load_anim.str[(i / 1000) % load_anim.size]);

                tensor_2d_view(&view, desc->test_inputs, i);

                network_feedforward(nn, out, &view);

                tensor_2d_view(&view, desc->test_outputs, i);
                if (tensor_argmax(out).x == tensor_argmax(&view).x) {
                    num_correct += 1;
                }
            }

            accuracy = (f32)num_correct / desc->test_inputs->shape.depth;

            printf("Test Accuracy: %f\n", accuracy);
        }

        if (desc->epoch_callback) {
            network_epoch_info info = {
                .epoch = epoch,

                .test_accuracy = accuracy
            };

            desc->epoch_callback(&info);
        }
    }

    // Unhides cursor
    printf("\e[?25h");

    os_thread_pool_destroy(tpool);

    mga_scratch_release(scratch);
}

static u64 _network_max_layer_size(const network* nn) {
    u64 max_layer_size = 0;
    for (u32 i = 0; i < nn->num_layers; i++) {
        tensor_shape s = nn->layers[i]->shape;

        u64 size = (u64)s.width * s.height * s.depth;

        if (size > max_layer_size) {
            max_layer_size = size;
        }
    }

    return max_layer_size;
} 


/*
Sample Summary:

-------------------------
    Network (5 layers)

type        shape
----        -----
input       (784, 1, 1)
dense       (64, 1, 1)
activation  (64, 1, 1)
dense       (10, 1, 1)
activation  (10, 1, 1)

-------------------------
*/
void network_summary(const network* nn) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8 header = str8_pushf(scratch.arena, "Network (%u layers)", nn->num_layers);

    // Storing strings in a list first to get good spacing in the console
    // +2 is for column name and "---" separator
    string8* types = MGA_PUSH_ZERO_ARRAY(scratch.arena, string8, nn->num_layers + 2);
    string8* shapes = MGA_PUSH_ZERO_ARRAY(scratch.arena, string8, nn->num_layers + 2);

    types[0] = STR8("type");
    types[1] = STR8("----");

    shapes[0] = STR8("shape");
    shapes[1] = STR8("-----");

    for (u32 i = 0; i < nn->num_layers; i++) {
        types[i + 2] = layer_get_name(nn->layers[i]->type);

        tensor_shape s = nn->layers[i]->shape;
        string8 shape_str = str8_pushf(scratch.arena, "(%u %u %u)", s.width, s.height, s.depth);

        shapes[i + 2] = shape_str;
    }

    u64 max_type_width = types[0].size;
    u64 max_shape_width = shapes[0].size;

    for (u32 i = 0; i < nn->num_layers; i++) {
        if (types[i + 2].size > max_type_width) {
            max_type_width = types[i + 2].size;
        }

        if (shapes[i + 2].size > max_shape_width) {
            max_shape_width = shapes[i + 2].size;
        }
    }

    // Spacing before, between, and after items
    u64 row_width = 1 + max_type_width + 2 + max_shape_width + 1;
    row_width = MAX(row_width, header.size + 2);

    // For even spacing of the header
    if ((row_width - header.size) % 2 != 0) {
        row_width += 1;
    }
    
    // For newline
    row_width++;

    // Borders + border padding + header + layers + titles
    u32 num_rows = 2 + 2 + 1 + nn->num_layers + 2;

    string8 out = {
        .size = row_width * num_rows,
        .str = MGA_PUSH_ARRAY(scratch.arena, u8, row_width * num_rows)
    };

    memset(out.str, ' ', out.size);
    for (u32 y = 0; y < num_rows; y++) {
        out.str[row_width - 1 + y * row_width] = '\n';
    }

    // Borders
    memset(out.str, '-', row_width - 1);
    memset(out.str + (num_rows - 1) * row_width, '-', row_width - 1);

    // Header
    u32 header_spacing = (row_width - 1 - header.size) / 2;
    memcpy(out.str + row_width + header_spacing, header.str, header.size);

    u32 shape_start_x = 1 + max_type_width + 2;
    for (u32 i = 0; i < nn->num_layers + 2; i++) {
        // Start index into row
        u64 start_i = (i + 3) * row_width;

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
void network_save_layout(const network* nn, string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string8_list save_list = { 0 };

    // For spacing between layer_descs
    string8 new_line = STR8("\n");

    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_desc_save(scratch.arena, &save_list, &nn->layer_descs[i]);
        str8_list_push(scratch.arena, &save_list, new_line);
    }

    os_file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

string8 network_get_tpn_header(void) {
    return _tpn_header;
}

/*
File Format (*.tpn):

Header
Network Layout (tpl)
Tensor List of layer params
*/
void network_save(const network* nn, string8 file_name) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    string8 layout_str = { 0 };

    {
        mga_temp scratch2 = mga_scratch_get(&scratch.arena, 1);

        string8_list layout_list = { 0 };
        for (u32 i = 0; i < nn->num_layers; i++) {
            layer_desc_save(scratch.arena, &layout_list, &nn->layer_descs[i]);
        }

        string8 full_layout_str = str8_concat(scratch2.arena, layout_list);
        layout_str = str8_remove_space(scratch.arena, full_layout_str);

        mga_scratch_release(scratch2);
    }


    tensor_list param_list = { 0 };
    for (u32 i = 0; i < nn->num_layers; i++) {
        layer_save(scratch.arena, &param_list, nn->layers[i], i);
    }

    string8 param_str = tensor_list_to_str(scratch.arena, &param_list);

    string8_list save_list = { 0 };
    str8_list_push(scratch.arena, &save_list, _tpn_header);
    str8_list_push(scratch.arena, &save_list, layout_str);
    str8_list_push(scratch.arena, &save_list, param_str);

    os_file_write(file_name, save_list);

    mga_scratch_release(scratch);
}

