#include <stdio.h>
#include <stdlib.h>

#include <turbospork/turbospork.h>

int main(int argc, char** argv) {
    ts_arena* perm_arena = ts_arena_create(TS_MiB(64), TS_MiB(1), TS_TRUE);

    ts_string8 a = TS_STR8_LIT("Hello World");
    printf("'%.*s'\n", TS_STR8_FMT(a));

    ts_string8 b = { 0 };
    if (argc > 1) {
        b = ts_str8_from_cstr((ts_u8*)argv[1]);
    }

    ts_string8 c = ts_str8_copy(perm_arena, b);

    printf("b (%p): '%.*s', c (%p): '%.*s'\n", b.str, TS_STR8_FMT(b), c.str, TS_STR8_FMT(c));
    printf("%d %d\n", ts_str8_equals(a, b), ts_str8_equals(b, c));

    ts_string8 d = ts_str8_substr(a, 0, 5);
    ts_string8 e = ts_str8_substr_size(a, 6, 5);
    printf("'%.*s' '%.*s'\n", TS_STR8_FMT(d), TS_STR8_FMT(e));

    ts_string8_list list = { 0 };
    ts_str8_list_add(perm_arena, &list, TS_STR8_LIT("String 0"));
    ts_str8_list_add(perm_arena, &list, TS_STR8_LIT("String 1"));
    ts_str8_list_add(perm_arena, &list, TS_STR8_LIT("String 2"));
    
    ts_string8_concat_desc concat_desc = {
        TS_STR8_LIT("{ "),
        TS_STR8_LIT(", "),
        TS_STR8_LIT(" }")
    };
    ts_string8 f = ts_str8_concat(perm_arena, &list, &concat_desc);

    printf("'%.*s'\n", TS_STR8_FMT(f));

    ts_arena_destroy(perm_arena);

    return 0;
}

