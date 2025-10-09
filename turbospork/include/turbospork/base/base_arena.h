
#define TS_ARENA_ALIGN sizeof(void*)
#define TS_ARENA_NUM_SCRATCH 2

#define TS_ARENA_SCRATCH_RESERVE TS_MiB(64)
#define TS_ARENA_SCRATCH_COMMIT TS_KiB(64)

typedef struct ts_arena {
    struct ts_arena* current;
    struct ts_arena* prev;

    ts_u64 reserve_size;
    ts_u64 commit_size;
    ts_b32 growable;

    ts_u64 base_pos;
    ts_u64 pos;
    ts_u64 commit_pos;
} ts_arena;

typedef struct {
    ts_arena* arena;
    ts_u64 start_pos;
} ts_arena_temp;

#define TS_PUSH_STRUCT(arena, T) (T*)ts_arena_push((arena), sizeof(T), TS_FALSE)
#define TS_PUSH_STRUCT_NZ(arena, T) (T*)ts_arena_push((arena), sizeof(T), TS_TRUE)
#define TS_PUSH_ARRAY(arena, T, n) (T*)ts_arena_push((arena), sizeof(T) * (n), TS_FALSE)
#define TS_PUSH_ARRAY_NZ(arena, T, n) (T*)ts_arena_push((arena), sizeof(T) * (n), TS_TRUE)

ts_arena* ts_arena_create(ts_u64 reserve_size, ts_u64 commit_size, ts_b32 growable);
void ts_arena_destroy(ts_arena* arena);
ts_u64 ts_arena_get_pos(ts_arena* arena);
void* ts_arena_push(ts_arena* arena, ts_u64 size, ts_b32 non_zero);
void ts_arena_pop(ts_arena* arena, ts_u64 size);
void ts_arena_pop_to(ts_arena* arena, ts_u64 pos);

ts_arena_temp ts_arena_temp_begin(ts_arena* arena);
void ts_arena_temp_end(ts_arena_temp temp);

ts_arena_temp ts_arena_scratch_get(ts_arena** conflicts, ts_u32 num_conflicts);
void ts_arena_scratch_release(ts_arena_temp scratch);

