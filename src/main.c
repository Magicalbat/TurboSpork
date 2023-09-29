#include <stdio.h>

#include "base/base.h"

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

    mgp_init();
    mgp_set_title(MGP_STR8("Random Numbers"));
    mgp_set_win_size(300, 300);

    f32 xs[16] = {
        27, 36, 54, 2, 13, 26, 38, 4, 7, 9, 103, 13, 1, 2, 3, 4
    };
    f32 ys[16] = {
        4, 3, 2, 1, 13, 103, 9, 7, 4, 38, 26, 13, 2, 54, 36, 27
    };

    mgp_points(16, xs, ys);

    mgp_plot_show();

    mga_destroy(perm_arena);

    return 0;
}