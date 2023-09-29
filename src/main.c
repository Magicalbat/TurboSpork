#include <stdio.h>

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

    mgp_set_win_size(300, 300);
    mgp_view v = {
        .left = 0.0f, .right = 5.0f,
        .top = 30.0f, .bottom = 0.0f
    };
    mgp_set_view(v);

    float xs[16];
    float ys[16];
    for (int i = 0; i < 16; i++) {
        xs[i] = i % 2 == 0 ? 2.718f : 3.14159265358979f;
        ys[i] = 2.0f * i + 7.0f;
    }

    mgp_lines(16, xs, ys);

    mgp_plot_show();

    mga_destroy(perm_arena);

    return 0;
}