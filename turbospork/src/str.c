#include <string.h>
#include <stdio.h>

#include "base_defs.h"
#include "str.h"

ts_string8 ts_str8_from_range(ts_u8* start, ts_u8* end) {
    if (start == NULL || end == NULL) {
        fprintf(stderr, "Cannot creates ts_string8 from NULL ptr\n");
        return (ts_string8){ 0 };
    }

    return (ts_string8){ (ts_u64)(end - start), start };
}
ts_string8 ts_str8_from_cstr(ts_u8* cstr) {
    if (cstr == NULL) {
        fprintf(stderr, "Cannot create ts_string8 from NULL ptr\n");
        return (ts_string8){ 0 };
    }

    ts_u8* ptr = cstr;
    for(; *ptr != 0; ptr += 1);
    return ts_str8_from_range(cstr, ptr);
}

ts_string8 ts_str8_copy(mg_arena* arena, ts_string8 str) {
    ts_string8 out = { 
        .str = (ts_u8*)mga_push(arena, str.size),
        .size = str.size
    };

    memcpy(out.str, str.str, str.size);
    
    return out;
}
ts_u8* ts_str8_to_cstr(mg_arena* arena, ts_string8 str) {
    ts_u8* out = MGA_PUSH_ARRAY(arena, ts_u8, str.size + 1);
    
    memcpy(out, str.str, str.size);
    out[str.size] = '\0';

    return out;
}

ts_b32 ts_str8_equals(ts_string8 a, ts_string8 b) {
    if (a.size != b.size)
        return false;

    for (ts_u64 i = 0; i < a.size; i++)  {
        if (a.str[i] != b.str[i])
            return false;
    }
    
    return true;
}
ts_b32 ts_str8_contains(ts_string8 str, ts_string8 sub) {
    for (ts_u64 i = 0; i < str.size - sub.size + 1; i++) {
        ts_b32 contains = true;
        for (ts_u64 j = 0; j < sub.size; j++) {
            if (str.str[i + j] != sub.str[j]) {
                contains = false;
                break;
            }
        }

        if (contains) {
            return true;
        }
    }

    return false;
}
ts_b32 ts_str8_contains_char(ts_string8 str, ts_u8 c) {
    for (ts_u64 i = 0; i < str.size; i++) {
        if (str.str[i] == c)
            return true;
    }

    return false;
}

ts_b32 ts_str8_index_of(ts_string8 str, ts_string8 sub, ts_u64* index) {
    if (index == NULL) {
        fprintf(stderr, "Cannot put index of ts_string8 into NULL ptr\n");
        return false;
    }

    for (ts_u64 i = 0; i < str.size; i++) {
        if (ts_str8_equals(ts_str8_substr(str, i, i + sub.size), sub)) {
            *index = i;

            return true;
        }
    }

    return false;
}

ts_b32 ts_str8_index_of_char(ts_string8 str, ts_u8 c, ts_u64* index) {
    if (index == NULL) {
        fprintf(stderr, "Cannot put index of ts_string8 into NULL ptr\n");
        return false;
    }

    for (ts_u64 i = 0; i < str.size; i++) {
        if (str.str[i] == c) {
            *index = i;
            return true;
        }
    }

    return false;
}

ts_string8 ts_str8_substr(ts_string8 str, ts_u64 start, ts_u64 end) {
    ts_u64 end_clamped = TS_MIN(str.size, end);
    ts_u64 start_clamped = TS_MIN(start, end_clamped);
    return (ts_string8){ end_clamped - start_clamped, str.str + start_clamped };
}
ts_string8 ts_str8_substr_size(ts_string8 str, ts_u64 start, ts_u64 size) {
    return ts_str8_substr(str, start, start + size);
}

ts_string8 ts_str8_remove_space(mg_arena* arena, ts_string8 str) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_string8 stripped = {
        .size = str.size,
        .str = MGA_PUSH_ZERO_ARRAY(scratch.arena, ts_u8, str.size)
    };

    ts_u64 s_i = 0;
    for (ts_u64 i = 0; i < str.size; i++) {
        ts_u8 c = str.str[i];
        
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            stripped.str[s_i++] = str.str[i];
        } else {
            stripped.size--;
        }
    }

    ts_string8 out = ts_str8_copy(arena, stripped);

    mga_scratch_release(scratch);

    return out;
}

void ts_str8_list_push_existing(ts_string8_list* list, ts_string8 str, ts_string8_node* node) {
    if (list == NULL || node == NULL) {
        fprintf(stderr, "Cannot push node to string list: list or node is NULL\n");

        return;
    }

    node->str = str;
    TS_SLL_PUSH_BACK(list->first, list->last, node);
    list->node_count++;
    list->total_size += str.size;
}
void ts_str8_list_push(mg_arena* arena, ts_string8_list* list, ts_string8 str) {
    if (list == NULL) {
        fprintf(stderr, "Cannot push string to list: list is NULL\n");

        return;
    }

    ts_string8_node* node = MGA_PUSH_ZERO_STRUCT(arena, ts_string8_node);
    ts_str8_list_push_existing(list, str, node);
}

ts_string8 ts_str8_concat(mg_arena* arena, ts_string8_list list) {
    ts_string8 out = {
        .str = MGA_PUSH_ZERO_ARRAY(arena, ts_u8, list.total_size),
        .size = list.total_size
    };

    ts_u8* ptr = out.str;

    for (ts_string8_node* node = list.first; node != NULL; node = node->next) {
        memcpy(ptr, node->str.str, node->str.size);
        ptr += node->str.size;
    }

    return out;
}

