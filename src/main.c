#include <stdio.h>

#include "base/base.h"
#include "os/os.h"

#include "tensor/tensor.h"

#include "mg/mg_arena.h"
#include "mg/mg_plot.h"

void mga_on_error(mga_error err) {
    fprintf(stderr, "MGA Error %u: %s\n", err.code, err.msg);
}

void print_tensor(const tensorf* t) {
    printf("[ ");

    u64 size = (u64)t->shape.width * t->shape.height * t->shape.depth;
    for (u64 i = 0; i < size; i++) {
        printf("%f ", t->data[i]);
    }
    
    printf("]\n");
}

int main(void) {
    mga_desc desc = {
        .desired_max_size = MGA_MiB(64),
        .desired_block_size = MGA_KiB(256),
        .error_callback = mga_on_error
    };
    mg_arena* perm_arena = mga_create(&desc);

    u32 size = 2 * 3 * 4;
    tensorf* t1 = tensorf_create(perm_arena, (tensor_shape){ 2, 3, 4 });
    tensorf* t2 = tensorf_create(perm_arena, (tensor_shape){ 2, 3, 4 });
    for (u32 i = 0; i < size; i++) {
        t1->data[i] = i;
        t2->data[i] = size - i - 1;
    }

    printf("t1: ");
    print_tensor(t1);
    printf("t2: ");
    print_tensor(t2);

    tensorf* t3 = tensorf_add(perm_arena, t1, t2);

    printf("\n\nt3: ");
    print_tensor(t3);

    mga_destroy(perm_arena);

    return 0;
}
