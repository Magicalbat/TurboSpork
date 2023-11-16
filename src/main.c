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

#define MNIST_DIGIT_WIDTH 28
#define MNIST_DIGIT_HEIGHT 28
void draw_mnist_digit(f32* digit_data);

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
    mga_scratch_set_desc(&desc);
    mg_arena* perm_arena = mga_create(&desc);

    u64 seeds[2] = { 0 };
    os_get_entropy(seeds, sizeof(seeds));
    prng_seed(seeds[0], seeds[1]);

    dataset data = { 0 };

    tensor_list mnist = tensor_list_load(perm_arena, STR8("data/mnist.tpt"));
    data.train_imgs = tensor_list_get(&mnist, STR8("training_images"));
    data.train_labels = tensor_list_get(&mnist, STR8("training_labels"));
    data.test_imgs = tensor_list_get(&mnist, STR8("testing_images"));
    data.test_labels = tensor_list_get(&mnist, STR8("testing_labels"));

    network* nn = network_load_layout(perm_arena, STR8("networks/mnist_layout.tpl"), true);
    //network* nn = network_load(perm_arena, STR8("networks/mnist.tpn"), true);
    network_summary(nn);

    network_train_desc train_desc = {
        .epochs = 8,
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

        .train_inputs = data.train_imgs,
        .train_outputs = data.train_labels,

        .accuracy_test = true,
        .test_inputs = data.test_imgs,
        .test_outputs = data.test_labels
    };

    os_time_init();

    u64 start = os_now_microseconds();

    network_train(nn, &train_desc);

    u64 end = os_now_microseconds();

    printf("Train Time: %f\n", (f64)(end - start) / 1e6);

    //network_save(nn, STR8("networks/mnist.tpn"));

    network_delete(nn);

    mga_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(f32* digit_data) {
    mgp_init();
    mgp_set_title(MGP_STR8("MNIST Digit"));
    mgp_set_win_size(600, 600);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    u32 size = MNIST_DIGIT_WIDTH * MNIST_DIGIT_HEIGHT;
    mgp_vec4f* colors = MGA_PUSH_ARRAY(scratch.arena, mgp_vec4f, size);
    mgp_rectf* rects = MGA_PUSH_ARRAY(scratch.arena, mgp_rectf, size);

    for (u32 x = 0; x < MNIST_DIGIT_WIDTH; x++) {
        for (u32 y = 0; y < MNIST_DIGIT_HEIGHT; y++) {
            u32 i = x + y * MNIST_DIGIT_WIDTH;

            f32 c = digit_data[i];
            colors[i] = (mgp_vec4f){ c, c, c, 1.0f };

            rects[i] = (mgp_rectf){
                x, MNIST_DIGIT_HEIGHT - 1 - y, 1, 1
            };
        }
    }

    mgp_rects_ex(size, rects, (mgp_vec4f){ 0 }, colors, (mgp_string8){ 0 });

    mgp_plot_show();

    mga_scratch_release(scratch);
}
