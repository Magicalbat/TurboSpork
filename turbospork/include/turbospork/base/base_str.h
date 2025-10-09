
typedef struct {
    ts_u8* str;
    ts_u64 size;
} ts_string8;

typedef struct ts_string8_node {
    ts_string8 str;
    struct ts_string8_node* next;
} ts_string8_node;

typedef struct {
    ts_string8_node* first;
    ts_string8_node* last;

    ts_u32 count;
    ts_u64 total_size;
} ts_string8_list;

typedef struct {
    ts_string8 begin;
    ts_string8 delim;
    ts_string8 end;
} ts_string8_concat_desc;

#define TS_STR8_LIT(s) (ts_string8){ (ts_u8*)(s), sizeof(s) - 1 }

ts_string8 ts_str8_from_cstr(ts_u8* cstr);
ts_u8* ts_str8_to_cstr(ts_arena* arena, ts_string8 str);
ts_string8 ts_str8_copy(ts_arena* arena, ts_string8 src);

ts_b32 ts_str8_equals(ts_string8 a, ts_string8 b);

ts_string8 ts_str8_substr(ts_string8 base, ts_u64 start, ts_u64 end);
ts_string8 ts_str8_substr_size(ts_string8 base, ts_u64 start, ts_u64 size);

ts_string8 ts_str8_concat(
    ts_arena* arena,
    const ts_string8_list* list,
    const ts_string8_concat_desc* desc
);

void ts_str8_list_add_existing(ts_string8_list* list, ts_string8_node* node);
void ts_str8_list_add(ts_arena* arena, ts_string8_list* list, ts_string8 str);

