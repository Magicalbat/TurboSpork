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

    u32 size = 2 * 3 * 4;
    tensorf* t = tensorf_create_alloc(perm_arena, (tensor_shape){ .width = 2, .height = 3, .depth = 4 }, size * 2);
    for (u32 i = 0; i < size; i++) {
        t->data[i] = i;
    }

    printf("orig: [\n\t");
    for (u32 i = 0; i < size; i++) {
        printf("%f, ", t->data[i]);
    }
    printf("\n]\n");

    tensorf* copy = tensorf_copy(perm_arena, t, true);
    printf("copy alloc: %lu\n", copy->alloc);
    printf("copy: [\n\t");
    for (u32 i = 0; i < size; i++) {
        printf("%f, ", copy->data[i]);
    }
    printf("\n]\n");

    tensorf view_2d = { 0 };
    tensorf_2d_view(&view_2d, copy, 3);
    
    printf("copy: [\n\t");
    for (u32 i = 0; i < view_2d.alloc; i++) {
        printf("%f, ", view_2d.data[i]);
    }
    printf("\n]\n");

    tensorf_fill(&view_2d, 2.0f);
    
    printf("copy: [\n\t");
    for (u32 i = 0; i < size; i++) {
        printf("%f, ", copy->data[i]);
    }
    printf("\n]\n");

    mga_destroy(perm_arena);

    return 0;
}
