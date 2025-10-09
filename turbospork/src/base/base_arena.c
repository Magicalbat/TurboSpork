
ts_arena* ts_arena_create(ts_u64 reserve_size, ts_u64 commit_size, ts_b32 growable) {
    ts_u32 page_size = ts_plat_page_size();

    reserve_size = TS_ALIGN_UP_POW2(reserve_size, page_size);
    commit_size = TS_ALIGN_UP_POW2(commit_size, page_size);

    ts_arena* arena = ts_plat_mem_reserve(reserve_size);

    if (ts_plat_mem_commit(arena, commit_size) == TS_FALSE) {
        arena = TS_NULL;
    }

    if (arena == TS_NULL) {
        fprintf(stderr, "Fatal error: unable to commit memory for arena\n");
        ts_plat_exit(1);
    }

    // TODO: ASAN stuff for memory

    arena->current = arena;
    arena->prev = TS_NULL;

    arena->reserve_size = reserve_size;
    arena->commit_size = commit_size;
    arena->growable = growable;

    arena->base_pos = 0;
    arena->pos = sizeof(ts_arena);
    arena->commit_pos = commit_size;

    return arena;
}

void ts_arena_destroy(ts_arena* arena) {
    ts_arena* current = arena->current;

    while (current != TS_NULL) {
        ts_arena* prev = current->prev;
        ts_plat_mem_release(current);

        current = prev;
    }
}

ts_u64 ts_arena_get_pos(ts_arena* arena) {
    return arena->current->base_pos + arena->current->pos;
}

void* ts_arena_push(ts_arena* arena, ts_u64 size, ts_b32 non_zero) {
    void* out = TS_NULL;

    ts_arena* current = arena->current;

    ts_u64 pos_aligned = TS_ALIGN_UP_POW2(current->pos, TS_ARENA_ALIGN);
    out = (ts_u8*)current + pos_aligned;
    ts_u64 new_pos = pos_aligned + size;

    if (new_pos > current->reserve_size) {
        out = TS_NULL;

        if (arena->growable) {
            ts_u64 reserve_size = arena->reserve_size;
            ts_u64 commit_size = arena->commit_size;

            if (size + sizeof(ts_arena) > reserve_size) {
                reserve_size = TS_ALIGN_UP_POW2(size + sizeof(ts_arena), TS_ARENA_ALIGN);
            }

            ts_arena* new_arena = ts_arena_create(reserve_size, commit_size, TS_TRUE);
            new_arena->base_pos = current->base_pos + current->reserve_size;

            ts_arena* prev_cur = current;
            current = new_arena;
            current->prev = prev_cur;
            arena->current = current;

            pos_aligned = TS_ALIGN_UP_POW2(current->pos, TS_ARENA_ALIGN);
            out = (ts_u8*)current + pos_aligned;
            new_pos = pos_aligned + size;
        }
    }

    if (new_pos > current->commit_pos) {
        ts_u64 new_commit_pos = new_pos;
        new_commit_pos += current->commit_size - 1;
        new_commit_pos -= new_commit_pos % current->commit_size;
        new_commit_pos = TS_MIN(new_commit_pos, current->reserve_size);

        ts_u64 commit_size = new_commit_pos - current->commit_pos;

        ts_u8* commit_pointer = (ts_u8*)current + current->commit_pos;

        if (ts_plat_mem_commit(commit_pointer, commit_size) == TS_FALSE) {
            out = TS_NULL;
        } else {
            current->commit_pos = new_commit_pos;
        }
    }

    if (out == TS_NULL) {
        fprintf(stderr, "Fatal error: failed to allocate memory on arena\n");
        ts_plat_exit(1);
    }

    current->pos = new_pos;

    if (!non_zero) {
        TS_MEM_ZERO(out, size);
    }

    return out;
}

void ts_arena_pop(ts_arena* arena, ts_u64 size) {
    size = TS_MIN(size, ts_arena_get_pos(arena));

    ts_arena* current = arena->current;
    while (current != TS_NULL && size > current->pos) {
        ts_arena* prev = current->prev;

        size -= current->pos;
        ts_plat_mem_release(current);

        current = prev;
    }

    arena->current = current;
    size = TS_MIN(current->pos - sizeof(ts_arena), size);
    current->pos -= size;
}

void ts_arena_pop_to(ts_arena* arena, ts_u64 pos) {
    ts_u64 cur_pos = ts_arena_get_pos(arena);
    
    pos = TS_MIN(pos, cur_pos);

    ts_arena_pop(arena, cur_pos - pos);
}

ts_arena_temp ts_arena_temp_begin(ts_arena* arena) {
    return (ts_arena_temp) {
        .arena = arena,
        .start_pos = ts_arena_get_pos(arena)
    };
}

void ts_arena_temp_end(ts_arena_temp temp) {
    ts_arena_pop_to(temp.arena, temp.start_pos);
}

static TS_THREAD_LOCAL ts_arena* scratch_arenas[2] = { 0 };

ts_arena_temp ts_arena_scratch_get(ts_arena** conflicts, ts_u32 num_conflicts) {
    ts_i32 scratch_index = -1;

    for (ts_i32 i = 0; i < 2; i++) {
        ts_b32 conflict_found = TS_FALSE;

        for (ts_u32 j = 0; j < num_conflicts; j++) {
            if (scratch_arenas[i] == conflicts[j]) {
                conflict_found = TS_TRUE;
                break;
            }
        }

        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        return (ts_arena_temp){ 0 };
    }

    if (scratch_arenas[scratch_index] == NULL) {
        scratch_arenas[scratch_index] = ts_arena_create(
            TS_ARENA_SCRATCH_RESERVE,
            TS_ARENA_SCRATCH_COMMIT,
            TS_FALSE
        );
    }

    return ts_arena_temp_begin(scratch_arenas[scratch_index]);
}

void ts_arena_scratch_release(ts_arena_temp scratch) {
    ts_arena_temp_end(scratch);
}

