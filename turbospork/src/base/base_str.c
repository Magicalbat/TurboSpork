
ts_string8 ts_str8_from_cstr(ts_u8* cstr) {
    ts_u8* start = cstr;

    while (*(cstr++));

    return (ts_string8) {
        .str = start,
        .size = (ts_u64)(cstr - start)
    };
}

ts_u8* ts_str8_to_cstr(ts_arena* arena, ts_string8 str) {
    ts_u8* out = TS_PUSH_ARRAY_NZ(arena, ts_u8, str.size);

    memcpy(out, str.str, str.size);
    out[str.size] = '\0';

    return out;
}

ts_string8 ts_str8_copy(ts_arena* arena, ts_string8 src) {
    ts_string8 out = {
        .str = TS_PUSH_ARRAY_NZ(arena, ts_u8, src.size),
        .size = src.size
    };

    memcpy(out.str, src.str, src.size);

    return out;
}

ts_b32 ts_str8_equals(ts_string8 a, ts_string8 b) {
    if (a.size != b.size) {
        return TS_FALSE;
    }

    for (ts_u64 i = 0; i < a.size; i++) {
        if (a.str[i] != b.str[i]) {
            return TS_FALSE;
        }
    }

    return TS_TRUE;
}

ts_string8 ts_str8_substr(ts_string8 base, ts_u64 start, ts_u64 end) {
    end = TS_MIN(base.size, end);
    start = TS_MIN(end, start);

    return (ts_string8) {
        .str = base.str + start,
        .size = end - start
    };
}

ts_string8 ts_str8_substr_size(ts_string8 base, ts_u64 start, ts_u64 size) {
    start = TS_MIN(base.size, start);
    size = TS_MIN(size, base.size - start);

    return (ts_string8) {
        .str = base.str + start,
        .size = size
    };
}

ts_string8 ts_str8_concat(
    ts_arena* arena,
    const ts_string8_list* list,
    const ts_string8_concat_desc* desc
) {
    if (list->count == 0) {
        return (ts_string8) { 0 };
    }

    ts_u64 total_size = list->total_size +
        desc->begin.size + desc->delim.size * (list->count - 1) + desc->end.size;

    ts_string8 out = (ts_string8){
        .str = TS_PUSH_ARRAY_NZ(arena, ts_u8, total_size),
        .size = total_size
    };

    memcpy(out.str, desc->begin.str, desc->begin.size);

    ts_u64 pos = desc->begin.size;

    ts_string8_node* node = list->first;
    for (ts_u32 i = 0; i < list->count && node != TS_NULL; i++, node = node->next) {
        memcpy(out.str + pos, node->str.str, node->str.size);
        pos += node->str.size;

        memcpy(out.str + pos, desc->delim.str, desc->delim.size);
        pos += desc->delim.size;
    }

    memcpy(out.str + pos, desc->end.str, desc->end.size);

    return out;
}

void ts_str8_list_add_existing(ts_string8_list* list, ts_string8_node* node) {
    list->count++;
    list->total_size += node->str.size;
    
    TS_SLL_PUSH_BACK(list->first, list->last, node);
}

void ts_str8_list_add(ts_arena* arena, ts_string8_list* list, ts_string8 str) {
    ts_string8_node* node = TS_PUSH_STRUCT(arena, ts_string8_node);
    ts_str8_list_add_existing(list, node);
}

