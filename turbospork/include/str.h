#ifndef STR_H
#define STR_H

#include <stdarg.h>

#include "base_defs.h"
#include "mg/mg_arena.h"

// This is heavily based on the string 
// header in Mr 4th programming series: 
// https://github.com/Mr-4th-Programming/mr4th/blob/main/src/base/base_string.h

typedef struct {
    ts_u64 size;
    ts_u8* str;
} ts_string8;

typedef struct {
    ts_u64 size;
    ts_u16* str;
} ts_string16;

typedef struct {
    ts_u64 size;
    ts_u32* str;
} ts_string32;

typedef struct ts_string8_node {
    struct ts_string8_node* next;
    ts_string8 str;
} ts_string8_node;

typedef struct {
    ts_string8_node* first;
    ts_string8_node* last;
    ts_u64 node_count;
    ts_u64 total_size;
} ts_string8_list; 

typedef struct {
    ts_string8 pre;
    ts_string8 inbetween;
    ts_string8 post;
} ts_string8_join;

typedef struct {
    ts_u32 code_point;
    ts_u32 size;
} ts_string_decode;

#define TS_STR8(s) ((ts_string8){ sizeof(s)-1, (ts_u8*)s })

ts_string8 ts_str8_from_range(ts_u8* start, ts_u8* end);
ts_string8 ts_str8_from_cstr(ts_u8* cstr);

ts_string8 ts_str8_copy(mg_arena* arena, ts_string8 str);
ts_u8*     ts_str8_to_cstr(mg_arena* arena, ts_string8 str);

ts_b32 ts_str8_equals(ts_string8 a, ts_string8 b);
ts_b32 ts_str8_contains(ts_string8 a, ts_string8 b);
ts_b32 ts_str8_contains_char(ts_string8 str, ts_u8 c);

ts_b32 ts_str8_index_of(ts_string8 str, ts_string8 sub, ts_u64* index);
ts_b32 ts_str8_index_of_char(ts_string8 str, ts_u8 c, ts_u64* index);

ts_string8 ts_str8_substr(ts_string8 str, ts_u64 start, ts_u64 end);
ts_string8 ts_str8_substr_size(ts_string8 str, ts_u64 start, ts_u64 size);

// Removes all ' ', '\t', '\n', and '\r'
ts_string8 ts_str8_remove_space(mg_arena* arena, ts_string8 str);

void ts_str8_list_push_existing(ts_string8_list* list, ts_string8 str, ts_string8_node* node);
void ts_str8_list_push(mg_arena* arena, ts_string8_list* list, ts_string8 str);

ts_string8 ts_str8_concat(mg_arena* arena, ts_string8_list list);
ts_string8 ts_str8_join(mg_arena* arena, ts_string8_list list, ts_string8_join join);

ts_string8 ts_str8_pushfv(mg_arena* arena, const char* fmt, va_list args);
ts_string8 ts_str8_pushf(mg_arena* arena, const char* fmt, ...);

ts_string_decode ts_str_decode_utf8(ts_u8* str, ts_u32 cap);
ts_u32           ts_str_encode_utf8(ts_u8* dst, ts_u32 code_point);

ts_string_decode ts_str_decode_utf16(ts_u16* str, ts_u32 cap);
ts_u32           ts_str_encode_utf16(ts_u16* dst, ts_u32 code_point);

ts_string32 ts_str32_from_str8(mg_arena* arena, ts_string8 str);
ts_string8  ts_str8_from_str32(mg_arena* arena, ts_string32 str);
ts_string16 ts_str16_from_str8(mg_arena* arena, ts_string8 str);
ts_string8  ts_str8_from_str16(mg_arena* arena, ts_string16 str);

#endif // BASE_STR_H
