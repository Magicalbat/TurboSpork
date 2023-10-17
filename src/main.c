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
void draw_mnist_digit(const tensor* digit);

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

    dataset data = { 0 };

#if 1

    tensor_list mnist = tensor_list_load(perm_arena, STR8("data/mnist.tpt"));
    data.train_imgs = tensor_list_get(&mnist, STR8("training_images"));
    data.train_labels = tensor_list_get(&mnist, STR8("training_labels"));
    data.test_imgs = tensor_list_get(&mnist, STR8("testing_images"));
    data.test_labels = tensor_list_get(&mnist, STR8("testing_labels"));

#else 

    // Just zeros and ones
    {
        mga_temp scratch = mga_scratch_get(NULL, 0);
        
        tensor_list mnist = tensor_list_load(scratch.arena, STR8("data/mnist.tpt"));
        tensor* train_imgs = tensor_list_get(&mnist, STR8("training_images"));
        tensor* train_labels = tensor_list_get(&mnist, STR8("training_labels"));
        tensor* test_imgs = tensor_list_get(&mnist, STR8("testing_images"));
        tensor* test_labels = tensor_list_get(&mnist, STR8("testing_labels"));

        u32 train_size = 0;
        for (u32 i = 0; i < train_imgs->shape.depth; i++) {
            u32 j = i * train_labels->shape.width;
            if (train_labels->data[j] == 1.0f || train_labels->data[j + 1] == 1.0f) {
                train_size += 1.0f;
            }
        }

        data.train_imgs = tensor_create(perm_arena, (tensor_shape){ train_imgs->shape.width, 1, train_size });
        data.train_labels = tensor_create(perm_arena, (tensor_shape){ train_labels->shape.width, 1, train_size });

        u64 imgs_i = 0;
        u64 labels_i = 0;

        for (u64 i = 0; i < train_imgs->shape.depth; i++) {
            u64 j = i * train_labels->shape.width;
            if (train_labels->data[j] == 1.0f || train_labels->data[j + 1] == 1.0f) {
                memcpy(&data.train_imgs->data[imgs_i], &train_imgs->data[i * train_imgs->shape.width], sizeof(f32) * train_imgs->shape.width);
                memcpy(&data.train_labels->data[labels_i], &train_labels->data[i * train_labels->shape.width], sizeof(f32) * train_labels->shape.width);

                imgs_i += train_imgs->shape.width;
                labels_i += train_labels->shape.width;
            }
        }

        u32 test_size = 0;
        for (u32 i = 0; i < test_imgs->shape.depth; i++) {
            u32 j = i * test_labels->shape.width;
            if (test_labels->data[j] == 1.0f || test_labels->data[j + 1] == 1.0f) {
                test_size += 1.0f;
            }
        }

        data.test_imgs = tensor_create(perm_arena, (tensor_shape){ test_imgs->shape.width, 1, test_size });
        data.test_labels = tensor_create(perm_arena, (tensor_shape){ test_labels->shape.width, 1, test_size });

        imgs_i = 0;
        labels_i = 0;

        for (u64 i = 0; i < test_imgs->shape.depth; i++) {
            u64 j = i * test_labels->shape.width;
            if (test_labels->data[j] == 1.0f || test_labels->data[j + 1] == 1.0f) {
                memcpy(&data.test_imgs->data[imgs_i], &test_imgs->data[i * test_imgs->shape.width], sizeof(f32) * test_imgs->shape.width);
                memcpy(&data.test_labels->data[labels_i], &test_labels->data[i * test_labels->shape.width], sizeof(f32) * test_labels->shape.width);

                imgs_i += test_imgs->shape.width;
                labels_i += test_labels->shape.width;
            }
        }

        mga_scratch_release(scratch);
    }
#endif

    // Initial memory is not used
    tensor* img = tensor_create(perm_arena, (tensor_shape){ 1, 1, 1 });
    tensor* label = tensor_create(perm_arena, (tensor_shape){ 1, 1, 1 });
    tensor_2d_view(img, data.train_imgs, 0);
    tensor_2d_view(label, data.train_labels, 0);

    b32 training_mode = true;
    layer_desc layer_descs[] = {
        (layer_desc){
            .type = LAYER_DENSE,
            .training_mode = training_mode,
            .dense = (layer_dense_desc){
                .in_size = 784,
                .out_size = 30
            }
        },
        (layer_desc){
            .type = LAYER_ACTIVATION,
            .training_mode = training_mode,
            .activation = (layer_activation_desc) {
                .type = ACTIVATION_SIGMOID,
                .shape = (tensor_shape){ 30, 1, 1 }
            }
        },
        (layer_desc){
            .type = LAYER_DENSE,
            .training_mode = training_mode,
            .dense = (layer_dense_desc){
                .in_size = 30,
                .out_size = 10
            }
        },
        (layer_desc){
            .type = LAYER_ACTIVATION,
            .training_mode = training_mode,
            .activation = (layer_activation_desc) {
                .type = ACTIVATION_SIGMOID,
                .shape = (tensor_shape){ 10, 1, 1 }
            }
        },
    };
    network* nn = network_create(perm_arena, sizeof(layer_descs) / sizeof(layer_desc), layer_descs);

    tensor* in_out = tensor_copy(perm_arena, img, false);

    network_feedforward(nn, in_out, in_out);
    
    printf("[ ");
    for (u32 i = 0; i < 10; i++) {
        printf("%f ", in_out->data[i]);
    }
    printf("]\n");

    network_train_desc train_desc = {
        .epochs = 5,
        .batch_size = 10,

        .cost = COST_QUADRATIC,
        .optim_desc = (optimizer_desc){
            .type = OPTIMIZER_SGD,
            .learning_rate = 0.05,
        },
        
        .train_inputs = data.train_imgs,
        .train_outputs = data.train_labels,

        .accuracy_test = false
    };
    network_train(nn, &train_desc);
    printf("\n");

    in_out = tensor_copy(perm_arena, img, false);

    network_feedforward(nn, in_out, in_out);

    printf("[ ");
    for (u32 i = 0; i < 10; i++) {
        printf("%f ", in_out->data[i]);
    }
    printf("]\n");

    // test accuracy
    {
        u32 num_correct = 0;
        tensor* out = tensor_create(perm_arena, (tensor_shape){ 10, 1, 1 });
        tensor view = { 0 };

        for (u32 i = 0; i < data.test_imgs->shape.depth; i++) {
            tensor_2d_view(&view, data.test_imgs, i);

            network_feedforward(nn, out, &view);

            tensor_2d_view(&view, data.test_labels, i);
            if (tensor_argmax(out).x == tensor_argmax(&view).x) {
                num_correct += 1;
            }
        }

        printf("accuracy: %f\n", (f32)num_correct / data.test_imgs->shape.depth);
    }
    
    mga_destroy(perm_arena);

    return 0;
}

void draw_mnist_digit(const tensor* digit) {
    mgp_init();
    mgp_set_title(MGP_STR8("MNIST Digit"));
    mgp_set_win_size(600, 600);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    u32 size = MNIST_DIGIT_WIDTH * MNIST_DIGIT_HEIGHT;
    mgp_vec4f* colors = MGA_PUSH_ARRAY(scratch.arena, mgp_vec4f, size);
    mgp_rectf* rects = MGA_PUSH_ARRAY(scratch.arena, mgp_rectf, size);

    for (u32 x = 0; x < 28; x++) {
        for (u32 y = 0; y < 28; y++) {
            u32 i = x + y * 28;

            f32 c = digit->data[i];
            colors[i] = (mgp_vec4f){ c, c, c, 1.0f };

            rects[i] = (mgp_rectf){
                x, 27 - y, 1, 1
            };
        }
    }

    mgp_rects_ex(size, rects, (mgp_vec4f){ 0 }, colors, (mgp_string8){ 0 });

    mgp_plot_show();

    mga_scratch_release(scratch);

}

