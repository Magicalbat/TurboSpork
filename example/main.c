#include <stdio.h>
#include <stdlib.h>

#include <turbospork/turbospork.h>

void print_dist(
    ts_f32* nums, ts_u32 count,
    ts_u32 height, ts_u32 width,
    ts_f32 min, ts_f32 max
);

int main(void) {
    ts_arena* perm_arena = ts_arena_create(TS_MiB(64), TS_MiB(1), TS_TRUE);

    ts_u64 seeds[2] = { 0 };
    ts_plat_get_entropy(seeds, sizeof(seeds));
    ts_prng_context prng = { seeds[0], seeds[1] };

    ts_u32 count = 1000000;
    ts_f32* nums = TS_PUSH_ARRAY(perm_arena, ts_f32, count);

    for (ts_u32 i = 0; i < count; i++) {
        nums[i] = ts_prng_std_norm_r(&prng);
    }

    ts_u32 counts[] = { 100, 1000, 10000, 100000, 1000000 };

    for (ts_u32 i = 0; i < sizeof(counts) / sizeof(counts[0]); i++) {
        printf("%u: \n", counts[i]);
        print_dist(nums, counts[i], 8, 50, -3.0f, 3.0f);
        printf("\n");
    }
    
    ts_arena_destroy(perm_arena);

    return 0;
}

void print_dist(
    ts_f32* nums, ts_u32 count,
    ts_u32 height, ts_u32 width,
    ts_f32 min, ts_f32 max
) {
    ts_arena_temp scratch = ts_arena_scratch_get(NULL, 0);

    ts_f32* heights = TS_PUSH_ARRAY(scratch.arena, ts_f32, width);
    ts_f32 sector_size = (max - min) / (ts_f32)width;

    for (ts_u32 i = 0; i < width; i++) {
        ts_f32 sector_start = min + sector_size * (ts_f32)i;
        ts_f32 sector_end = min + sector_size * (ts_f32)(i + 1);

        for (ts_u32 j = 0; j < count; j++) {
            if (sector_start <= nums[j] && nums[j] < sector_end) {
                heights[i] += 1.0f;
            }
        }

        heights[i] *= (ts_f32)height / (ts_f32)count;
    }

    ts_f32 max_height = 0.0f;
    for (ts_u32 i = 0; i < width; i++) {
        if (heights[i] > max_height)
            max_height = heights[i];
    }

    for (ts_u32 i = 0; i < width; i++) {
        heights[i] *= (ts_f32)height / max_height;
    }

    ts_string8 height_gradient = TS_STR8_LIT(" _.-oO#");

    ts_u8* line = TS_PUSH_ARRAY(scratch.arena, ts_u8, width);

    for (ts_i32 y = (ts_i32)height-1; y >= 0; y--) {
        for (ts_u32 x = 0; x < width; x++) {
            ts_f32 cur_height = TS_CLAMP(heights[x] - (ts_f32)y, 0.0f, 0.99f);
            ts_u32 char_index = (ts_u32)floorf(
                cur_height * (ts_f32)height_gradient.size
            );

            line[x] = height_gradient.str[char_index];
        }

        printf("%.*s\n", (int)width, (char*)line);
    }

    ts_arena_scratch_release(scratch);
}

