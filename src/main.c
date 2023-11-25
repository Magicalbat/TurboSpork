#include <stdio.h>
#include <string.h>

#include "base/base.h"
#include "os/os.h"

#include "layers/layers.h"
#include "costs/costs.h"
#include "optimizers/optimizers.h"
#include "network/network.h"

#include "tensor/tensor.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

void draw_mnist_digit(f32* digit_data, u32 width, u32 height);

typedef struct {
    tensor* train_imgs;
    tensor* train_labels;
    tensor* test_imgs;
    tensor* test_labels;
} dataset;

void mga_on_error(mga_error err) {
    fprintf(stderr, "MGA Error %u: %s\n", err.code, err.msg);
}

int main(void) {
    mga_desc desc = {
        .desired_max_size = MGA_MiB(256),
        .desired_block_size = MGA_MiB(1),
        .error_callback = mga_on_error
    };
    mg_arena* perm_arena = mga_create(&desc);
    mga_scratch_set_desc(&desc);

    u64 seeds[2] = { 0 };
    os_get_entropy(seeds, sizeof(seeds));
    prng_seed(seeds[0], seeds[1]);

    dataset data = { 0 };

    tensor_list mnist = tensor_list_load(perm_arena, STR8("data/mnist.tpt"));
    data.train_imgs = tensor_list_get(&mnist, STR8("training_images"));
    data.train_labels = tensor_list_get(&mnist, STR8("training_labels"));
    data.test_imgs = tensor_list_get(&mnist, STR8("testing_images"));
    data.test_labels = tensor_list_get(&mnist, STR8("testing_labels"));

    //network* nn = network_load_layout(perm_arena, STR8("networks/mnist_layout.tpl"), true);

    layer_desc descs[] = {
        {
            .type = LAYER_INPUT,
            .input.shape = { 28, 28, 1 },
        },
        {
            .type = LAYER_CONV_2D,
            .conv_2d = (layer_conv_2d_desc){
                .num_filters = 16,
                .kernel_size = { 3, 3, 1 },
                .padding = true,
            }
        },
        {
            .type = LAYER_ACTIVATION,
            .activation.type = ACTIVATION_RELU,
        },
        {
            .type = LAYER_POOLING_2D,
            .pooling_2d = (layer_pooling_2d_desc){
                .pool_size = { 2, 2, 1 },
                .type = POOLING_MAX
            }
        },
        {
            .type = LAYER_CONV_2D,
            .conv_2d = (layer_conv_2d_desc){
                .num_filters = 32,
                .kernel_size = { 3, 3, 1 },
                .padding = true,
            }
        },
        {
            .type = LAYER_ACTIVATION,
            .activation.type = ACTIVATION_RELU,
        },
        {
            .type = LAYER_POOLING_2D,
            .pooling_2d = (layer_pooling_2d_desc){
                .pool_size = { 2, 2, 1 },
                .type = POOLING_MAX
            }
        },
        {
            .type = LAYER_FLATTEN
        },
        {
            .type = LAYER_DENSE,
            .dense.size = 10
        },
        {
            .type = LAYER_ACTIVATION,
            .activation.type = ACTIVATION_SOFTMAX
        }
    };

    network* nn = network_load_layout(perm_arena, STR8("networks/mnist_conv.tpl"), true);
    //network* nn = network_create(perm_arena, sizeof(descs) / sizeof(layer_desc), descs, true);
    //network_save_layout(nn, STR8("networks/mnist_conv.tpl"));

    network_summary(nn);

    network_train_desc train_desc = {
        .epochs = 16,
        .batch_size = 100,

        .num_threads = 8,

        .cost = COST_CATEGORICAL_CROSS_ENTROPY,
        .optim = (optimizer){
            .type = OPTIMIZER_ADAM,
            .learning_rate = 0.0002f,

            .adam = (optimizer_adam){
                .beta1 = 0.9f,
                .beta2 = 0.999f,
                .epsilon = 1e-8f
            }
        },

        .save_interval = 0,
        //.save_path = STR8("training_nets/network_"),

        .train_inputs = data.train_imgs,
        .train_outputs = data.train_labels,

        .accuracy_test = true,
        .test_inputs = data.test_imgs,
        .test_outputs = data.test_labels
    };

    tensor in_view = { 0 };
    tensor_2d_view(&in_view, data.train_imgs, 0);
    tensor* in_out = tensor_copy(perm_arena, &in_view, false);

    network_feedforward(nn, in_out, in_out);

    for (u32 i = 0; i < 10; i++) {
        printf("%f ", in_out->data[i]);
    }
    printf("\n");

    os_time_init();

    u64 start = os_now_microseconds();

    //network_train(nn, &train_desc);

    u64 end = os_now_microseconds();

    printf("Train Time: %f\n", (f64)(end - start) / 1e6);

    //network_save(nn, STR8("networks/mnist.tpn"));

    network_delete(nn);

    mga_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* digit_data, u32 width, u32 height) {
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

            f32 c = digit_data[i];
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
