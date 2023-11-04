#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "base/base_defs.h"
#include "base/base_str.h"

string8 str8_from_range(u8* start, u8* end) {
    return (string8){ (u64)(end - start), start };
}
string8 str8_from_cstr(u8* cstr) {
    u8* ptr = cstr;
    for(; *ptr != 0; ptr += 1);
    return str8_from_range(cstr, ptr);
}

string8 str8_copy(mg_arena* arena, string8 str) {
    string8 out = { 
        .str = (u8*)mga_push(arena, str.size),
        .size = str.size
    };

    memcpy(out.str, str.str, str.size);
    
    return out;
}
u8* str8_to_cstr(mg_arena* arena, string8 str) {
    u8* out = MGA_PUSH_ARRAY(arena, u8, str.size + 1);
    
    memcpy(out, str.str, str.size);
    out[str.size] = '\0';

    return out;
}

b32 str8_equals(string8 a, string8 b) {
    if (a.size != b.size)
        return false;

    for (u64 i = 0; i < a.size; i++)  {
        if (a.str[i] != b.str[i])
            return false;
    }
    
    return true;
}
b32 str8_contains(string8 a, string8 b) {
    for (u64 i = 0; i < a.size - b.size + 1; i++) {
        b8 contains = true;
        for (u64 j = 0; j < b.size; j++) {
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
b32 str8_contains_char(string8 str, u8 c) {
    for (u64 i = 0; i < str.size; i++) {
        if (str.str[i] == c)
            return true;
    }
    return false;
}

b32 str8_index_of(string8 str, u8 c, u64* index) {
    for (u64 i = 0; i < str.size; i++) {
        if (str.str[i] == c) {
            *index = i;
            return true;
        }
    }

    return false;
}

string8 str8_substr(string8 str, u64 start, u64 end) {
    u64 end_clamped = MIN(str.size, end);
    u64 start_clamped = MIN(start, end_clamped);
    return (string8){ end_clamped - start_clamped, str.str + start_clamped };
}
string8 str8_substr_size(string8 str, u64 start, u64 size) {
    return str8_substr(str, start, start + size);
}

#define _WHITESPACE(c) ((c) == ' ' || (c) == '\t' || (c) == '\n')
string8 str8_remove_space(mg_arena* arena, string8 str) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    string8 stripped = {
        .size = str.size,
        .str = MGA_PUSH_ZERO_ARRAY(scratch.arena, u8, str.size)
    };

    u64 s_i = 0;
    for (u64 i = 0; i < str.size; i++) {
        u8 c = str.str[i];
        
        if (c != ' ' && c != '\t' && c != '\n') {
            stripped.str[s_i++] = str.str[i];
        } else {
            stripped.size--;
        }
    }

    string8 out = str8_copy(arena, stripped);

    mga_scratch_release(scratch);

    return out;
}

void str8_list_push_existing(string8_list* list, string8 str, string8_node* node) {
    node->str = str;
    SLL_PUSH_BACK(list->first, list->last, node);
    list->node_count++;
    list->total_size += str.size;
}
void str8_list_push(mg_arena* arena, string8_list* list, string8 str) {
    string8_node* node = MGA_PUSH_ZERO_STRUCT(arena, string8_node);
    str8_list_push_existing(list, str, node);
}

string8 str8_concat(mg_arena* arena, string8_list list) {
    string8 out = {
        .str = (u8*)mga_push(arena, list.total_size),
        .size = list.total_size
    };

    u8* ptr = out.str;

    for (string8_node* node = list.first; node != NULL; node = node->next) {
        memcpy(ptr, node->str.str, node->str.size);
        ptr += node->str.size;
    }

    return out;
}
string8 str8_join(mg_arena* arena, string8_list list, string8_join join) {
    u64 out_size = join.pre.size + join.inbetween.size * (list.node_count - 1) + list.total_size + join.post.size + 1;
    
    string8 out = {
        .str = (u8*)mga_push(arena, out_size),
        .size = out_size
    };

    memcpy(out.str, join.pre.str, join.pre.size);

    u8* ptr = out.str + join.pre.size;

    for (string8_node* node = list.first; node != NULL; node = node->next) {
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

string8 str8_pushfv(mg_arena* arena, const char* fmt, va_list args) {
    va_list args2;
    va_copy(args2, args);

    u64 init_size = 1024;
    u8* buffer = MGA_PUSH_ARRAY(arena, u8, init_size);
    u64 size = vsnprintf((char*)buffer, init_size, fmt, args);

    string8 out = { 0 };
    if (size < init_size) {
        mga_pop(arena, init_size - size - 1);
        out = (string8){ size, buffer };
    } else {
        // NOTE: This path may not work
        mga_pop(arena, init_size);
        u8* fixed_buff = MGA_PUSH_ARRAY(arena, u8, size + 1);
        u64 final_size = vsnprintf((char*)fixed_buff, size + 1, fmt, args);
        out = (string8){ final_size, fixed_buff };
    }

    va_end(args2);

    return out;
}

string8 str8_pushf(mg_arena* arena, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    string8 out = str8_pushfv(arena, fmt, args);

    va_end(args);

    return out;
}

// https://github.com/skeeto/branchless-utf8/blob/master/utf8.h
// https://github.com/Mr-4th-Programming/mr4th/blob/main/src/base/base_string.cpp
string_decode str_decode_utf8(u8* str, u32 cap) {
    static u8 lengths[] = {
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
    static u8 first_byte_mask[] = { 0, 0x7F, 0x1F, 0x0F, 0x07 };
    static u8 final_shift[] = { 0, 18, 12, 6, 0 };

    string_decode out = { .size=0 };

    if (cap > 0) {
        out.code_point = '#';
        out.size = 1;
        
        u32 len = lengths[str[0] >> 3];
        if (len > 0 && len <= cap) {
            u32 code_point = (str[0] & first_byte_mask[len]) << 18;
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

u32 str_encode_utf8(u8* dst, u32 code_point) {
    u32 size = 0;

    if (code_point < (1 << 8)) {
        dst[0] = (u8)code_point;
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
string_decode str_decode_utf16(u16* str, u32 cap) {
    string_decode out = { '#', 1 };
    u16 x = str[0];

    if (x < 0xd800 || x >= 0xdfff) {
        out.code_point = x;
    } else if (cap >= 2) {
        u16 y = str[1];
        if (x >= 0xd800 && x <= 0xdbff && y >= 0xdc00 && y <= 0xdfff) {
            u16 x2 = x - 0xd800;
            u16 y2 = y - 0xdc00;
            out.code_point = ((x2 << 10) | y2) + 0x10000;
            out.size = 2;
        }
    }

    return out;
}
u32 str_encode_utf16(u16* dst, u32 code_point) {
    u32 size = 0;

    if (code_point < 0x010000) {
        dst[0] = (u16)code_point;
        size = 1;
    } else {
        u32 u_p = code_point - 0x10000;
        dst[0] = 0xd800 + (u_p >> 10);
        dst[0] = 0xdc00 + (u_p & 0x3ff);
        size = 2;
    }

    return size;
}

string32 str32_from_str8(mg_arena* arena, string8 str) {
    u32* buff = MGA_PUSH_ARRAY(arena, u32, str.size + 1);

    u32* ptr_out = buff;
    u8* ptr = str.str;
    u8* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        string_decode decode = str_decode_utf8(ptr, (u32)(ptr_end - ptr));

        *ptr_out = decode.code_point;

        ptr += decode.size;
        ptr_out += 1;
    }

    *ptr_out = 0;

    u64 alloc_count = str.size + 1;
    u64 string_count = (u64)(ptr_out - buff);
    u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (string32){ .str = buff, .size = string_count };
}
string8 str8_from_str32(mg_arena* arena, string32 str) {
    u8* buff = MGA_PUSH_ARRAY(arena, u8, str.size * 4 + 1);

    u8* ptr_out = buff;
    u32* ptr = str.str;
    u32* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        u32 encode_size = str_encode_utf8(ptr_out, *ptr);

        ptr_out += encode_size;
        ptr += 1;
    }

    *ptr_out = 0;

    u64 alloc_count = str.size * 4 + 1;
    u64 string_count = (u64)(ptr_out - buff);
    u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (string8){ .str = buff, .size = string_count };
}
string16 str16_from_str8(mg_arena* arena, string8 str) {
    u16* buff = MGA_PUSH_ARRAY(arena, u16, str.size * 2 + 1);

    u16* ptr_out = buff;
    u8* ptr = str.str;
    u8* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        string_decode decode = str_decode_utf8(ptr, (u32)(ptr_end - ptr));
        u32 encode_size = str_encode_utf16(ptr_out, decode.code_point);

        ptr += decode.size;
        ptr_out += encode_size;
    }

    *ptr_out = 0;

    u64 alloc_count = str.size * 2 + 1;
    u64 string_count = (u64)(ptr_out - buff);
    u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (string16){ .str = buff, .size = string_count };

}
string8 str8_from_str16(mg_arena* arena, string16 str) {
    u8* buff = MGA_PUSH_ARRAY(arena, u8, str.size * 4 + 1);

    u8* ptr_out = buff;
    u16* ptr = str.str;
    u16* ptr_end = str.str + str.size;
    for(;ptr < ptr_end;){
        string_decode decode = str_decode_utf16(ptr, (u32)(ptr_end - ptr));
        u16 encode_size = str_encode_utf8(ptr_out, decode.code_point);

        ptr_out += encode_size;
        ptr += decode.size;
    }

    *ptr_out = 0;

    u64 alloc_count = str.size * 4 + 1;
    u64 string_count = (u64)(ptr_out - buff);
    u64 unused_count = alloc_count - string_count - 1;
    mga_pop(arena, unused_count * (sizeof(*buff)));

    return (string8){ .str = buff, .size = string_count };
}
