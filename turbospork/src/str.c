#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "base_defs.h"
#include "str.h"

ts_string8 ts_str8_from_range(ts_u8* start, ts_u8* end) {
    return (ts_string8){ (ts_u64)(end - start), start };
}
ts_string8 ts_str8_from_cstr(ts_u8* cstr) {
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
ts_b32 ts_str8_contains(ts_string8 a, ts_string8 b) {
    for (ts_u64 i = 0; i < a.size - b.size + 1; i++) {
        ts_b32 contains = true;
        for (ts_u64 j = 0; j < b.size; j++) {
            if (a.str[i + j] != b.str[j]) {
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
    for (ts_u64 i = 0; i < str.size; i++) {
        if (ts_str8_equals(ts_str8_substr(str, i, i + sub.size), sub)) {
            *index = i;

            return true;
        }
    }

    return false;
}

ts_b32 ts_str8_index_of_char(ts_string8 str, ts_u8 c, ts_u64* index) {
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
    node->str = str;
    TS_SLL_PUSH_BACK(list->first, list->last, node);
    list->node_count++;
    list->total_size += str.size;
}
void ts_str8_list_push(mg_arena* arena, ts_string8_list* list, ts_string8 str) {
    ts_string8_node* node = MGA_PUSH_ZERO_STRUCT(arena, ts_string8_node);
    ts_str8_list_push_existing(list, str, node);
}

ts_string8 ts_str8_concat(mg_arena* arena, ts_string8_list list) {
    ts_string8 out = {
        .str = (ts_u8*)mga_push(arena, list.total_size),
        .size = list.total_size
    };

    ts_u8* ptr = out.str;

    for (ts_string8_node* node = list.first; node != NULL; node = node->next) {
        memcpy(ptr, node->str.str, node->str.size);
        ptr += node->str.size;
    }

    return out;
}
ts_string8 ts_str8_join(mg_arena* arena, ts_string8_list list, ts_string8_join join) {
    ts_u64 out_size = join.pre.size + join.inbetween.size * (list.node_count - 1) + list.total_size + join.post.size + 1;
    
    ts_string8 out = {
        .str = (ts_u8*)mga_push(arena, out_size),
        .size = out_size
    };

    memcpy(out.str, join.pre.str, join.pre.size);

    ts_u8* ptr = out.str + join.pre.size;

    for (ts_string8_node* node = list.first; node != NULL; node = node->next) {
        if (node != list.first) {
            memcpy(ptr, join.inbetween.str, join.inbetween.size);
            ptr += join.inbetween.size;
        }

        memcpy(ptr, node->str.str, node->str.size);
        ptr += node->str.size;
    }

    memcpy(ptr, join.post.str, join.post.size);
    ptr += join.post.size;

    *ptr = 0;

    return out;
}

ts_string8 ts_str8_pushfv(mg_arena* arena, const char* fmt, va_list args) {
    va_list args2;
    va_copy(args2, args);

    ts_u64 init_size = 1024;
    ts_u8* buffer = MGA_PUSH_ARRAY(arena, ts_u8, init_size);
    ts_u64 size = vsnprintf((char*)buffer, init_size, fmt, args);

    ts_string8 out = { 0 };
    if (size < init_size) {
        mga_pop(arena, init_size - size - 1);
        out = (ts_string8){ size, buffer };
    } else {
        // NOTE: This path may not work
        mga_pop(arena, init_size);
        ts_u8* fixed_buff = MGA_PUSH_ARRAY(arena, ts_u8, size + 1);
        ts_u64 final_size = vsnprintf((char*)fixed_buff, size + 1, fmt, args);
        out = (ts_string8){ final_size, fixed_buff };
    }

    va_end(args2);

    return out;
}

ts_string8 ts_str8_pushf(mg_arena* arena, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    ts_string8 out = ts_str8_pushfv(arena, fmt, args);

    va_end(args);

    return out;
}

// https://github.com/skeeto/branchless-utf8/blob/master/utf8.h
// https://github.com/Mr-4th-Programming/mr4th/blob/main/src/base/base_string.cpp
ts_string_decode ts_str_decode_utf8(ts_u8* str, ts_u32 cap) {
    static ts_u8 lengths[] = {
        1, 1, 1, 1, // 000xx
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        0, 0, 0, 0, // 100xx
        0, 0, 0, 0,
        2, 2, 2, 2, // 110xx
        3, 3,       // 1110x
        4,          // 11110
        0,          // 11111
    };
    static ts_u8 first_byte_mask[] = { 0, 0x7F, 0x1F, 0x0F, 0x07 };
    static ts_u8 final_shift[] = { 0, 18, 12, 6, 0 };

    ts_string_decode out = { .size=0 };

    if (cap > 0) {
        out.code_point = '#';
        out.size = 1;
        
        ts_u32 len = lengths[str[0] >> 3];
        if (len > 0 && len <= cap) {
            ts_u32 code_point = (str[0] & first_byte_mask[len]) << 18;
            switch(len) {
                case 4: code_point |= (str[3] & 0b00111111) << 0;
                // fall through
                case 3: code_point |= (str[2] & 0b00111111) << 6;
                // fall through
                case 2: code_point |= (str[1] & 0b00111111) << 12;
                // fall through
                default: break;
            }
            code_point >>= final_shift[len];

            out.code_point = code_point;
            out.size = len;
        }
    }

    return out;
}

ts_u32 ts_str_encode_utf8(ts_u8* dst, ts_u32 code_point) {
    ts_u32 size = 0;

    if (code_point < (1 << 8)) {
        dst[0] = (ts_u8)code_point;
        size = 1;
    } else if (code_point < (1 << 11)) {
        dst[0] = 0b11000000 | (code_point >> 6);
        dst[1] = 0b10000000 | (code_point & 0b00111111);
        size = 2;
    } else if (code_point < (1 << 16)) {
        dst[0] = 0b11100000 | (code_point >> 12);
        dst[1] = 0b10000000 | ((code_point >> 6) & 0b00111111);
        dst[1] = 0b10000000 | (code_point & 0b00111111);
        size = 3;
    } else if (code_point < (1 << 21)) {
        dst[0] = 0b11110000 | (code_point >> 18);
        dst[1] = 0b10000000 | ((code_point >> 12) & 0b00111111);
        dst[2] = 0b10000000 | ((code_point >> 6) & 0b00111111);
        dst[3] = 0b10000000 | (code_point & 0b00111111);
        size = 4;
    } else {
        dst[0] = '#';
        size = 1;
    }

    return size;
}

// https://en.wikipedia.org/wiki/UTF-16
ts_string_decode ts_str_decode_utf16(ts_u16* str, ts_u32 cap) {
    ts_string_decode out = { '#', 1 };
    ts_u16 x = str[0];

    if (x < 0xd800 || x >= 0xdfff) {
        out.code_point = x;
    } else if (cap >= 2) {
        ts_u16 y = str[1];
        if (x >= 0xd800 && x <= 0xdbff && y >= 0xdc00 && y <= 0xdfff) {
            ts_u16 x2 = x - 0xd800;
            ts_u16 y2 = y - 0xdc00;
            out.code_point = ((x2 << 10) | y2) + 0x10000;
            out.size = 2;
        }
    }

    return out;
}
ts_u32 ts_str_encode_utf16(ts_u16* dst, ts_u32 code_point) {
    ts_u32 size = 0;

    if (code_point < 0x010000) {
        dst[0] = (ts_u16)code_point;
        size = 1;
    } else {
        ts_u32 u_p = code_point - 0x10000;
        dst[0] = 0xd800 + (u_p >> 10);
        dst[0] = 0xdc00 + (u_p & 0x3ff);
        size = 2;
    }

    return size;
}

ts_string32 ts_str32_from_ts_str8(mg_arena* arena, ts_string8 str) {
    ts_u32* buff = MGA_PUSH_ARRAY(arena, ts_u32, str.size + 1);

    ts_u32* ptr_out = buff;
    ts_u8* ptr = str.str;
    ts_u8* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        ts_string_decode decode = ts_str_decode_utf8(ptr, (ts_u32)(ptr_end - ptr));

        *ptr_out = decode.code_point;

        ptr += decode.size;
        ptr_out += 1;
    }

    *ptr_out = 0;

    ts_u64 alloc_count = str.size + 1;
    ts_u64 string_count = (ts_u64)(ptr_out - buff);
    ts_u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (ts_string32){ .str = buff, .size = string_count };
}
ts_string8 ts_str8_from_str32(mg_arena* arena, ts_string32 str) {
    ts_u8* buff = MGA_PUSH_ARRAY(arena, ts_u8, str.size * 4 + 1);

    ts_u8* ptr_out = buff;
    ts_u32* ptr = str.str;
    ts_u32* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        ts_u32 encode_size = ts_str_encode_utf8(ptr_out, *ptr);

        ptr_out += encode_size;
        ptr += 1;
    }

    *ptr_out = 0;

    ts_u64 alloc_count = str.size * 4 + 1;
    ts_u64 string_count = (ts_u64)(ptr_out - buff);
    ts_u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (ts_string8){ .str = buff, .size = string_count };
}
ts_string16 ts_str16_from_str8(mg_arena* arena, ts_string8 str) {
    ts_u16* buff = MGA_PUSH_ARRAY(arena, ts_u16, str.size * 2 + 1);

    ts_u16* ptr_out = buff;
    ts_u8* ptr = str.str;
    ts_u8* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        ts_string_decode decode = ts_str_decode_utf8(ptr, (ts_u32)(ptr_end - ptr));
        ts_u32 encode_size = ts_str_encode_utf16(ptr_out, decode.code_point);

        ptr += decode.size;
        ptr_out += encode_size;
    }

    *ptr_out = 0;

    ts_u64 alloc_count = str.size * 2 + 1;
    ts_u64 string_count = (ts_u64)(ptr_out - buff);
    ts_u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (ts_string16){ .str = buff, .size = string_count };

}
ts_string8 ts_str8_from_str16(mg_arena* arena, ts_string16 str) {
    ts_u8* buff = MGA_PUSH_ARRAY(arena, ts_u8, str.size * 4 + 1);

    ts_u8* ptr_out = buff;
    ts_u16* ptr = str.str;
    ts_u16* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        ts_string_decode decode = ts_str_decode_utf16(ptr, (ts_u32)(ptr_end - ptr));
        ts_u16 encode_size = ts_str_encode_utf8(ptr_out, decode.code_point);

        ptr_out += encode_size;
        ptr += decode.size;
    }

    *ptr_out = 0;

    ts_u64 alloc_count = str.size * 4 + 1;
    ts_u64 string_count = (ts_u64)(ptr_out - buff);
    ts_u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (ts_string8){ .str = buff, .size = string_count };
}
