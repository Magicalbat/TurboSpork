
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

ts_u32 ts_plat_page_size(void) {
    SYSTEM_INFO sysinfo = { 0 };
    GetSystemInfo(&sysinfo);

    return sysinfo.dwPageSize;
}

void* ts_plat_mem_reserve(ts_u64 size) {
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_READWRITE);
}

ts_b32 ts_plat_mem_commit(ts_u8* ptr, ts_u64 size) {
    void* ret = VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
    return ret != NULL;
}

ts_b32 ts_plat_mem_decommit(ts_u8* ptr, ts_u64 size) {
    return VirtualFree(ptr, size, MEM_DECOMMIT);
}

ts_b32 ts_plat_mem_release(ts_u8* ptr) {
    return VirtualFree(ptr, 0, MEM_RELEASE);
}


