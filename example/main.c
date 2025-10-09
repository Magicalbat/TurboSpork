#include <stdio.h>
#include <stdlib.h>

#include <turbospork/turbospork.h>

int main(void) {
    ts_arena* perm_arena = ts_arena_create(TS_MiB(64), TS_MiB(1), TS_TRUE);

    printf("%llu %llu\n", sizeof(ts_arena), ts_arena_get_pos(perm_arena));

    ts_u8* buf0 = TS_PUSH_ARRAY(perm_arena, ts_u8, TS_KiB(1));
    memset(buf0, 'a', TS_KiB(1));
    ts_u8* buf1 = TS_PUSH_ARRAY_NZ(perm_arena, ts_u8, TS_MiB(63));
    memset(buf1, 'b', TS_MiB(63));

    printf("%llu\n", ts_arena_get_pos(perm_arena));

    ts_u8* buf2 = TS_PUSH_ARRAY(perm_arena, ts_u8, TS_MiB(256));
    memset(buf2, 'c', TS_MiB(256));

    printf("%llu\n", ts_arena_get_pos(perm_arena));

    ts_u8* buf3 = TS_PUSH_ARRAY(perm_arena, ts_u8, TS_MiB(12));
    memset(buf3, 'd', TS_MiB(12));

    ts_u8* buf4 = TS_PUSH_ARRAY(perm_arena, ts_u8, TS_MiB(12));
    memset(buf4, 'e', TS_MiB(12));

    ts_arena_pop(perm_arena, TS_MiB(256));

    printf("%llu\n", ts_arena_get_pos(perm_arena));

    ts_arena_temp scratch = ts_arena_scratch_get(NULL, 0);
    ts_u8* buf5 = TS_PUSH_ARRAY(perm_arena, ts_u8, TS_MiB(16));
    memset(buf5, 'f', TS_MiB(16));

    ts_arena_scratch_release(scratch);

    ts_arena_destroy(perm_arena);

    return 0;
}

