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

    tensor_list mnist = tensor_list_load(perm_arena, STR8("data/mnist.tpt"));
    tensor* train_imgs = tensor_list_get(&mnist, STR8("training_images"));
    tensor* train_labels = tensor_list_get(&mnist, STR8("training_labels"));

    #if 0
    // Just zeros and ones
    tensor* easy_train_imgs = NULL;
    tensor* easy_train_labels = NULL;
    {
        mga_temp scratch = mga_scratch_get(NULL, 0);
        
        tensor_list mnist = tensor_list_load(scratch.arena, STR8("data/mnist.tpt"));
        tensor* train_imgs = tensor_list_get(&mnist, STR8("training_images"));
        tensor* train_labels = tensor_list_get(&mnist, STR8("training_labels"));

        u32 easy_size = 0;
        for (u32 i = 0; i < train_imgs->shape.depth; i++) {
            u32 j = i * train_labels->shape.width;
            if (train_labels->data[j] == 1.0f || train_labels->data[j + 1] == 1.0f) {
                easy_size += 1.0f;
            }
        }

        easy_train_imgs = tensor_create(perm_arena, (tensor_shape){ train_imgs->shape.width, 1, easy_size });
        easy_train_labels = tensor_create(perm_arena, (tensor_shape){ train_labels->shape.width, 1, easy_size });

        u64 imgs_i = 0;
        u64 labels_i = 0;

        for (u64 i = 0; i < train_imgs->shape.depth; i++) {
            u64 j = i * train_labels->shape.width;
            if (train_labels->data[j] == 1.0f || train_labels->data[j + 1] == 1.0f) {
                memcpy(&easy_train_imgs->data[imgs_i], &train_imgs->data[i * train_imgs->shape.width], sizeof(f32) * train_imgs->shape.width);
                memcpy(&easy_train_labels->data[labels_i], &train_labels->data[i * train_labels->shape.width], sizeof(f32) * train_labels->shape.width);

                imgs_i += train_imgs->shape.width;
                labels_i += train_labels->shape.width;
            }
        }

        mga_scratch_release(scratch);
    }
#endif

    // Initial memory is not used
    tensor* img = tensor_create(perm_arena, (tensor_shape){ 1, 1, 1 });
    tensor* label = tensor_create(perm_arena, (tensor_shape){ 1, 1, 1 });
    tensor_2d_view(img, train_imgs, 0);
    tensor_2d_view(label, train_labels, 0);

    b32 training_mode = true;
    layer_desc layer_descs[] = {
        (layer_desc){
            .type = LAYER_DENSE,
            .training_mode = training_mode,
            .dense = (layer_dense_desc){
                .in_size = 784,
                .out_size = 64
            }
        },
        (layer_desc){
            .type = LAYER_ACTIVATION,
            .training_mode = training_mode,
            .activation = (layer_activation_desc) {
                .type = ACTIVATION_RELU,
                .shape = (tensor_shape){ 64, 1, 1 }
            }
        },
        (layer_desc){
            .type = LAYER_DENSE,
            .training_mode = training_mode,
            .dense = (layer_dense_desc){
                .in_size = 64,
                .out_size = 10
            }
        },
        (layer_desc){
            .type = LAYER_ACTIVATION,
            .training_mode = training_mode,
            .activation = (layer_activation_desc) {
                .type = ACTIVATION_SOFTMAX,
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
        .epochs = 2,
        .batch_size = 20,

        .cost = COST_QUADRATIC,
        .optim_desc = (optimizer_desc){
            .type = OPTIMIZER_SGD,
            .learning_rate = 0.05,
        },
        
        .train_inputs = train_imgs,
        .train_outputs = train_labels,

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

