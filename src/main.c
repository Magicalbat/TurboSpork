#include <stdio.h>

#include "base/base.h"
#include "os/os.h"

#include "tensor/tensor.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

void mga_on_error(mga_error err) {
    fprintf(stderr, "MGA Error %u: %s\n", err.code, err.msg);
}

int main(void) {
    mga_desc desc = {
        .desired_max_size = MGA_MiB(64),
        .desired_block_size = MGA_KiB(256),
        .error_callback = mga_on_error
    };
    mg_arena* perm_arena = mga_create(&desc);

    tensorf* t = tensorf_create(perm_arena, (tensor_shape){ .width = 10, .height = 23 });

    printf("%d %d %d %llu\n", t->shape.width, t->shape.height, t->shape.depth, t->alloc);

    TENSORF_AT(t, 0, 0, 0) = 12.34f;

    printf("%f %f\n", t->data[0], t->data[1]);

    mga_destroy(perm_arena);

    return 0;
}
