#include <math.h>
#include <stdio.h>
#include <string.h>

#include "examples.h"

#include "turbospork.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

typedef ts_f32 f32;
typedef ts_u32 u32;
typedef ts_u64 u64;

static void draw_char(f32* data, u32 width, u32 height);

static void mga_on_error(mga_error err) {
    fprintf(stderr, "MGA Error %u: %s\n", err.code, err.msg);
}

void emnist_main(void) {
    mga_desc desc = {
        .desired_max_size = MGA_GiB(8),
        .desired_block_size = MGA_MiB(256),
        .error_callback = mga_on_error
    };
    mg_arena* perm_arena = mga_create(&desc);
    mga_scratch_set_desc(&desc);

    u64 seeds[2] = { 0 };
    ts_get_entropy(seeds, sizeof(seeds));
    ts_prng_seed(seeds[0], seeds[1]);

    ts_tensor_list emnist = ts_tensor_list_load(perm_arena, TS_STR8("data/emnist.tst"));
    ts_tensor* train_imgs = ts_tensor_list_get(&emnist, TS_STR8("train_inputs"));
    ts_tensor* train_labels = ts_tensor_list_get(&emnist, TS_STR8("train_labels"));
    ts_tensor* test_imgs = ts_tensor_list_get(&emnist, TS_STR8("test_inputs"));
    ts_tensor* test_labels = ts_tensor_list_get(&emnist, TS_STR8("test_labels"));

    //ts_network* nn = ts_network_load_layout(perm_arena, TS_STR8("networks/emnist.tsl"), true);
    ts_network* nn = ts_network_load(perm_arena, TS_STR8("training_nets/emnist_take2_0003.tsn"), true);

    ts_network_summary(nn);

    ts_network_train_desc train_desc = {
        .epochs = 32,
        .batch_size = 500,

        .num_threads = 8,

        .cost = TS_COST_CATEGORICAL_CROSS_ENTROPY,
        .optim = (ts_optimizer){
            .type = TS_OPTIMIZER_ADAM,
            .learning_rate = 0.0005f,

            .adam = (ts_optimizer_adam){
                .beta1 = 0.9f,
                .beta2 = 0.999f,
                .epsilon = 1e-7f
            }
        },

        .random_transforms = true,
        .transforms = (ts_network_transforms) {
            .min_translation = -2.0f,
            .max_translation =  2.0f,

            .min_scale = 0.9f,
            .max_scale = 1.1f,

            .min_angle = -3.14159265 / 16.0f,
            .max_angle =  3.14159265 / 16.0f,
        },

        .save_interval = 1,
        .save_path = TS_STR8("training_nets/emnist_take2_"),

        .train_inputs = train_imgs,
        .train_outputs = train_labels,

        .accuracy_test = true,
        .test_inputs = test_imgs,
        .test_outputs = test_labels
    };

    ts_time_init();

    u64 start = ts_now_usec();

    ts_network_train(nn, &train_desc);

    u64 end = ts_now_usec();

    printf("Train Time: %f\n", (ts_f64)(end - start) / 1e6);

    ts_network_delete(nn);

    mga_destroy(perm_arena);
}

/*
ts_tensor* out = ts_tensor_create(perm_arena, (ts_tensor_shape){62, 1, 1});

    for (ts_u32 i = 0; i < 0; i++) {
        ts_tensor slice = { 0 };
        ts_tensor_2d_view(&slice, test_labels, i);
        printf("%u\n", ts_tensor_argmax(&slice).x);

        ts_tensor_2d_view(&slice, test_imgs, i);
        ts_network_feedforward(nn, out, &slice);
        ts_f32* data = out->data;
        for (ts_u32 i = 0; i < 62; i++) {
            printf("%u ", isnan(data[i]));
        }
        printf("\n");

        slice.shape.width = 28;
        slice.shape.height = 28;
        ts_img_rotate_ip(&slice, &slice, TS_SAMPLE_BILINEAR, 3.14159265f / 2.0f);
        ts_img_scale_ip(&slice, &slice, TS_SAMPLE_BILINEAR, -1.0f, 1.0f);
        draw_char(slice.data, 28, 28);
    }


*/

static void draw_char(f32* data, u32 width, u32 height) {
    mgp_init();
    mgp_set_title(MGP_STR8("MNIST Digit"));
    mgp_set_win_size(600, 600);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    u32 size = width * height;
    mgp_vec4f* colors = MGA_PUSH_ARRAY(scratch.arena, mgp_vec4f, size);
    mgp_rectf* rects = MGA_PUSH_ARRAY(scratch.arena, mgp_rectf, size);

    for (u32 x = 0; x < width; x++) {
        for (u32 y = 0; y < height; y++) {
            u32 i = x + y * width;

            f32 c = data[i];
            if (c > 0.0)
                colors[i] = (mgp_vec4f){ c, 0, 0, 1.0f };
            else
                colors[i] = (mgp_vec4f){ 0, -c, 0, 1.0f };

            rects[i] = (mgp_rectf){
                x, height - 1 - y, 1, 1
            };
        }
    }

    mgp_rects_ex(size, rects, (mgp_vec4f){ 0 }, colors, (mgp_string8){ 0 });

    mgp_plot_show();

    mga_scratch_release(scratch);
}
