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

    mga_destroy(perm_arena);

    return 0;
}
