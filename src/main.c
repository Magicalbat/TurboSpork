#include <stdio.h>

#include "base/base.h"
#include "os/os.h"

#include "layers/layers.h"
#include "costs/costs.h"
#include "optimizers/optimizers.h"

#include "tensor/tensor.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

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

    tensor label = { 0 };
    tensor_2d_view(&label, train_labels, 0);

    printf("%u\n", tensor_argmax(&label).x);

    mgp_init();
    mgp_set_win_size(600, 600);

    mga_temp scratch = mga_scratch_get(NULL, 0);

    mgp_vec4f* colors = MGA_PUSH_ARRAY(scratch.arena, mgp_vec4f, train_imgs->shape.width);
    mgp_rectf* rects = MGA_PUSH_ARRAY(scratch.arena, mgp_rectf, train_imgs->shape.width);

    for (u32 x = 0; x < 28; x++) {
        for (u32 y = 0; y < 28; y++) {
            u32 i = x + y * 28;

            f32 c = train_imgs->data[i];
            colors[i] = (mgp_vec4f){ c, c, c, 1.0f };

            rects[i] = (mgp_rectf){
                x, 27 - y, 1, 1
            };
        }
    }

    mgp_rects_ex(train_imgs->shape.width, rects, (mgp_vec4f){ 0 }, colors, (mgp_string8){ 0 });

    mgp_plot_show();

    mga_scratch_release(scratch);

    mga_destroy(perm_arena);

    return 0;
}
