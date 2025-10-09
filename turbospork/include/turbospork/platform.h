
void ts_plat_exit(ts_i32 code);

ts_u32 ts_plat_page_size(void);

void* ts_plat_mem_reserve(ts_u64 size);

ts_b32 ts_plat_mem_commit(void* ptr, ts_u64 size);

ts_b32 ts_plat_mem_decommit(void* ptr, ts_u64 size);

ts_b32 ts_plat_mem_release(void* ptr);

