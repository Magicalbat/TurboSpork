#include <stdio.h>
#include <string.h>

#include "examples.h"

#include "turbospork.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

void draw_img(const ts_tensor* img);

static void mga_on_error(mga_error err) {
    fprintf(stderr, "MGA Error %u: %s\n", err.code, err.msg);
}

void cifar10_main(void) {
    mga_desc desc = {
        .desired_max_size = MGA_GiB(1),
        .desired_block_size = MGA_MiB(16),
        .error_callback = mga_on_error
    };
    mg_arena* perm_arena = mga_create(&desc);
    mga_scratch_set_desc(&desc);

    ts_u64 seeds[2] = { 0 };
    ts_get_entropy(seeds, sizeof(seeds));
    ts_prng_seed(seeds[0], seeds[1]);

    ts_tensor_list cifar10 = ts_tensor_list_load(perm_arena, TS_STR8("data/cifar10.tst"));
    ts_tensor* train_inputs = ts_tensor_list_get(&cifar10, TS_STR8("train_inputs"));
    ts_tensor* train_labels = ts_tensor_list_get(&cifar10, TS_STR8("train_labels"));
    ts_tensor* test_inputs = ts_tensor_list_get(&cifar10, TS_STR8("test_inputs"));
    ts_tensor* test_labels = ts_tensor_list_get(&cifar10, TS_STR8("test_labels"));
    
#if 0
    ts_tensor img = { 0 };
    ts_tensor label = { 0 };

    for (ts_u32 j = 0; j < 5; j++) {
        ts_tensor_2d_view(&img, train_inputs, j);
        ts_tensor_2d_view(&label, train_labels, j);

        img.shape = (ts_tensor_shape){ 32, 32, 3 };

        printf("[ ");
        for (ts_u32 i = 0; i < label.shape.width; i++) {
            printf("%f ", label.data[i]);
        }
        printf("]\n");

        draw_img(&img);
    }
#endif

    ts_network* nn = ts_network_load_layout(perm_arena, TS_STR8("networks/cifar10.tsl"), true);
    //ts_network* nn = ts_network_load(perm_arena, TS_STR8("training_nets/cifar_0001.tsn"), true);

    ts_network_summary(nn);

    for (int j = 0; j < 1; j++) {
        ts_tensor img = { 0 };
        ts_tensor label = { 0 };
        ts_tensor_2d_view(&img, test_inputs, j);
        ts_tensor_2d_view(&label, test_labels, j);

        ts_tensor* out = ts_tensor_create(perm_arena, (ts_tensor_shape){ 10, 1, 1 });
        ts_network_feedforward(nn, out, &img);

        for (ts_u32 i = 0; i < 10; i++) {
            printf("%f ", out->data[i]);
        }
        printf("\n");
    }

    ts_network_train_desc train_desc = {
        .epochs = 8,
        .batch_size = 100,

        .num_threads = 8,

        .cost = TS_COST_CATEGORICAL_CROSS_ENTROPY,
        .optim = (ts_optimizer){
            .type = TS_OPTIMIZER_ADAM,
            .learning_rate = 0.001f,

            .adam = (ts_optimizer_adam){
                .beta1 = 0.9f,
                .beta2 = 0.999f,
                .epsilon = 1e-7f
            }
        },

        //.save_interval = 1,
        //.save_path = TS_STR8("training_nets/cifar_"),

        .train_inputs = train_inputs,
        .train_outputs = train_labels,

        .accuracy_test = true,
        .test_inputs = test_inputs,
        .test_outputs = test_labels
    };

    ts_time_init();

    ts_u64 start = ts_now_usec();

    ts_network_train(nn, &train_desc);

    ts_u64 end = ts_now_usec();

    printf("Train Time: %f\n", (ts_f64)(end - start) / 1e6);

    for (ts_u32 j = 0; j < 10; j++) {
        ts_tensor img = { 0 };
        ts_tensor_2d_view(&img, train_inputs, j);
        ts_tensor* out = ts_tensor_create(perm_arena, (ts_tensor_shape){ 10, 1, 1 });
        ts_network_feedforward(nn, out, &img);

        for (ts_u32 i = 0; i < 10; i++) {
            printf("%f ", out->data[i]);
        }
        printf("\n");
    }

    ts_network_delete(nn);

    mga_destroy(perm_arena);
}

void draw_img(const ts_tensor* img) {
    mgp_init();
    mgp_set_title(MGP_STR8("cifar10 img"));
    mgp_set_win_size(600, 600);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    ts_u32 size = img->shape.width * img->shape.height;
    mgp_vec4f* colors = MGA_PUSH_ARRAY(scratch.arena, mgp_vec4f, size);
    mgp_rectf* rects = MGA_PUSH_ARRAY(scratch.arena, mgp_rectf, size);

    for (ts_u32 y = 0; y < img->shape.height; y++) {
        for (ts_u32 x = 0; x < img->shape.width; x++) {
            ts_u32 i = x + y * img->shape.width;

            if (img->shape.depth == 3) {
                colors[i] = (mgp_vec4f){
                    img->data[(0 * img->shape.height + y) * img->shape.width + x],
                    img->data[(1 * img->shape.height + y) * img->shape.width + x],
                    img->data[(2 * img->shape.height + y) * img->shape.width + x],
                    1.0f
                };
            } else {
                ts_f32 col = img->data[i];
                colors[i] = (mgp_vec4f){ col, col, col, 1.0f };
            }
            
            rects[i] = (mgp_rectf){
                x, img->shape.height - 1 - y, 1, 1
            };
        }
    }

    mgp_rects_ex(size, rects, (mgp_vec4f){ 0 }, colors, (mgp_string8){ 0 });

    mgp_plot_show();

    mga_scratch_release(scratch);

}

