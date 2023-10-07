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

    tensor* a = tensor_create(perm_arena, (tensor_shape){ 2, 3, 1 });
    tensor* b = tensor_create(perm_arena, (tensor_shape){ 3, 2, 1 });
    for (u32 i = 0; i < 2 * 3; i++) {
        a->data[i] = i;
        b->data[i] = i;
    }

    tensor_list list = { 0 };
    tensor_list_push(perm_arena, &list, a, STR8("tensor a"));
    tensor_list_push(perm_arena, &list, b, STR8("tensor b"));

    tensor_list_save(&list, STR8("out.tp"));

    mga_destroy(perm_arena);

    return 0;
}
