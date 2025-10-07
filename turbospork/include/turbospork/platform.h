
#if defined(__linux__)
#define TS_PLAT_LINUX
#elif defined(_WIN32)
#define TS_PLAT_WIN32
#endif

ts_u32 ts_plat_page_size(void);
void* ts_plat_mem_reserve(ts_u64 size);
ts_b32 ts_plat_mem_commit(ts_u8* ptr, ts_u64 size);
ts_b32 ts_plat_mem_decommit(ts_u8* ptr, ts_u64 size);
ts_b32 ts_plat_mem_release(ts_u8* ptr);

