#include "os.h"
#include "err.h"

#ifdef TS_PLATFORM_WIN32

#include <stdio.h>

#ifndef UNICODE
    #define UNICODE
#endif
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <timeapi.h>
#include <bcrypt.h>

static ts_u64 _ticks_per_sec = 1;

typedef struct {
    ts_u16* str;
    ts_u64 size;
} _string16;

static _string16 _utf16_from_utf8(mg_arena* arena, ts_string8 str) {
    mga_temp scratch = mga_scratch_get(&arena, 1);

    ts_u64 tmp_size = str.size * 2 + 1;
    ts_u16* tmp_out = MGA_PUSH_ZERO_ARRAY(scratch.arena, ts_u16, tmp_size);

    ts_i32 size_written = MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, (LPCCH)str.str, str.size, tmp_out, tmp_size);

    if (size_written == 0) {
        TS_ERR(TS_ERR_OS, "Failed to convert utf8 to utf16 for win32");

        mga_scratch_release(scratch);
        return (_string16){ 0 };
    }

    ts_u16* out = MGA_PUSH_ARRAY(scratch.arena, ts_u16, size_written);
    memcpy(out, tmp_out, sizeof(ts_u16) * size_written);

    mga_scratch_release(scratch);

    return (_string16){ .str = out, .size = size_written };
}

void ts_time_init(void) {
    LARGE_INTEGER perf_freq;
    if (QueryPerformanceFrequency(&perf_freq)) {
        _ticks_per_sec = ((ts_u64)perf_freq.HighPart << 32) | perf_freq.LowPart;
    } else {
        TS_ERR(TS_ERR_OS, "Failed to initialize time: could not get performance frequency");
    }
}

static ts_datetime _systime_to_datetime(SYSTEMTIME t){  
    return (ts_datetime){
        .sec   = (ts_u8 )t.wSecond,
        .min   = (ts_u8 )t.wMinute,
        .hour  = (ts_u8 )t.wHour,
        .day   = (ts_u8 )t.wDay,
        .month = (ts_u8 )t.wMonth,
        .year  = (ts_i32)t.wYear
    };
}

ts_datetime ts_now_localtime(void) {
    SYSTEMTIME t;
    GetLocalTime(&t);
    return _systime_to_datetime(t);
}

ts_u64 ts_now_usec(void) {
    ts_u64 out = 0;
    LARGE_INTEGER perf_count;
    if (QueryPerformanceCounter(&perf_count)) {
        ts_u64 ticks = ((ts_u64)perf_count.HighPart << 32) | perf_count.LowPart;
        out = ticks * 1000000 / _ticks_per_sec;
    } else {
        TS_ERR(TS_ERR_OS, "Failed to retrive time in micro seconds");
    }
    return out;
}
void ts_sleep_msec(ts_u32 t) {
    Sleep(t);
}

ts_string8 ts_file_read(mg_arena* arena, ts_string8 path) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    _string16 path16 = _utf16_from_utf8(scratch.arena, path);

    HANDLE file_handle = CreateFile(
        (LPCWSTR)path16.str,
        GENERIC_READ,
        0,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    mga_scratch_release(scratch);

    if (file_handle == INVALID_HANDLE_VALUE) {
        TS_ERR(TS_ERR_IO, "Failed to open file for reading");

        return (ts_string8){ 0 };
    }

    ts_string8 out = { 0 };

    DWORD high_size = 0;
    DWORD low_size = GetFileSize(file_handle, &high_size);
    ts_u64 total_size = ((ts_u64)high_size << 32) | low_size;

    mga_temp possible_temp = mga_temp_begin(arena);

    ts_u8* buffer = MGA_PUSH_ZERO_ARRAY(arena, ts_u8, total_size);

    ts_u64 total_read = 0;
    while (total_read < total_size) {
        ts_u64 to_read64 = total_size - total_read;
        DWORD to_read = to_read64 > ~(DWORD)(0) ? ~(DWORD)(0) : (DWORD)to_read64;

        DWORD bytes_read = 0;
        if (ReadFile(file_handle, buffer + total_read, to_read, &bytes_read, 0) == FALSE) {
            TS_ERR(TS_ERR_IO, "Failed to read from file");

            mga_temp_end(possible_temp);
            return (ts_string8){ 0 };
        }

        total_read += bytes_read;
    }

    out.size = total_size;
    out.str = buffer;

    CloseHandle(file_handle);
    return out;
}

ts_b32 _file_write_impl(HANDLE file_handle, ts_string8_list str_list) {
    for (ts_string8_node* node = str_list.first; node != NULL; node = node->next) {
        ts_u64 total_to_write = node->str.size;
        ts_u64 total_written = 0;

        while (total_written < total_to_write) {
            ts_u64 to_write64 = total_to_write - total_written;
            DWORD to_write = to_write64 > ~(DWORD)(0) ? ~(DWORD)(0) : (DWORD)to_write64;

            DWORD written = 0;
            if (WriteFile(file_handle, node->str.str + total_written, to_write, &written, 0) == FALSE) {
                return false;
            }

            total_written += written;
        }
    }

    return true;
}

ts_b32 ts_file_write(ts_string8 path, ts_string8_list str_list) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    _string16 path16 = _utf16_from_utf8(scratch.arena, path);

    HANDLE file_handle = CreateFile(
        (LPCWSTR)path16.str,
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    mga_scratch_release(scratch);

    if (file_handle == INVALID_HANDLE_VALUE) {
        TS_ERR(TS_ERR_IO, "Failed to open file for writing");

        return false;
    }

    ts_b32 out = true;

    if (!_file_write_impl(file_handle, str_list)) {
        TS_ERR(TS_ERR_IO, "Failed to write to file");

        out = false;
    }

    CloseHandle(file_handle);

    return out;

}
ts_file_stats ts_file_get_stats(ts_string8 path) {
    ts_file_stats stats = { 0 };

    mga_temp scratch = mga_scratch_get(NULL, 0);

    _string16 path16 = _utf16_from_utf8(scratch.arena, path);

    WIN32_FILE_ATTRIBUTE_DATA attribs = { 0 };
    if (GetFileAttributesEx((LPCWSTR)path16.str, GetFileExInfoStandard, &attribs) == FALSE) {
        TS_ERR(TS_ERR_IO, "Failed to get stats for file");
    }

    mga_scratch_release(scratch);

    stats.size = ((ts_u64)attribs.nFileSizeHigh << 32) | attribs.nFileSizeLow;

    if (attribs.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        stats.flags |= TS_FILE_IS_DIR;
    }

    SYSTEMTIME modify_sys_time = { 0 };
    FileTimeToSystemTime(&attribs.ftLastWriteTime, &modify_sys_time);
    stats.modify_time = _systime_to_datetime(modify_sys_time);

    return stats;
}

void ts_get_entropy(void* data, ts_u64 size) {
    BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

typedef struct _ts_mutex {
    CRITICAL_SECTION cs;
} ts_mutex;

ts_mutex* ts_mutex_create(mg_arena* arena) {
    ts_mutex* mutex = MGA_PUSH_ZERO_STRUCT(arena, ts_mutex);

    InitializeCriticalSection(&mutex->cs);

    return mutex;
}
void ts_mutex_destroy(ts_mutex* mutex) {
    DeleteCriticalSection(&mutex->cs);
}

// These two do not return any error information
// It looks like win32 will throw an exception if they fail
// For my purposes, I am always going to return true
ts_b32 ts_mutex_lock(ts_mutex* mutex) {
    EnterCriticalSection(&mutex->cs);

    return true;
}
ts_b32 ts_mutex_unlock(ts_mutex* mutex) {
    LeaveCriticalSection(&mutex->cs);

    return true;
}

typedef struct _ts_thread_pool {
    ts_u32 num_threads;
    HANDLE* threads;

    ts_u32 max_tasks;
    ts_u32 num_tasks;
    ts_thread_task* task_queue;

    CRITICAL_SECTION mutex; // I know that it is not technically a mutex on win32
    CONDITION_VARIABLE queue_cond_var;

    ts_u32 num_active;
    CONDITION_VARIABLE active_cond_var;
} ts_thread_pool;

static DWORD _thread_start(void* arg) {
    ts_thread_pool* tp = (ts_thread_pool*)arg;
    ts_thread_task task = { 0 };

    while (true) {
        EnterCriticalSection(&tp->mutex);
        while (tp->num_tasks == 0) {
            SleepConditionVariableCS(&tp->queue_cond_var, &tp->mutex, INFINITE);
        }

        tp->num_active++;
        task = tp->task_queue[0];
        for (ts_u32 i = 0; i < tp->num_tasks - 1; i++) {
            tp->task_queue[i] = tp->task_queue[i + 1];
        }
        tp->num_tasks--;

        LeaveCriticalSection(&tp->mutex);

        task.func(task.arg);

        EnterCriticalSection(&tp->mutex);

        tp->num_active--;
        if (tp->num_active == 0) {
            WakeConditionVariable(&tp->active_cond_var);
        }

        LeaveCriticalSection(&tp->mutex);
    }

    return 0;
}

ts_thread_pool* ts_thread_pool_create(mg_arena* arena, ts_u32 num_threads, ts_u32 max_tasks) {
    ts_thread_pool* tp = MGA_PUSH_ZERO_STRUCT(arena, ts_thread_pool);

    tp->max_tasks = max_tasks;
    tp->task_queue = MGA_PUSH_ZERO_ARRAY(arena, ts_thread_task, max_tasks);

    InitializeCriticalSection(&tp->mutex);
    InitializeConditionVariable(&tp->queue_cond_var);
    InitializeConditionVariable(&tp->active_cond_var);

    tp->num_threads = num_threads;
    tp->threads = MGA_PUSH_ZERO_ARRAY(arena, HANDLE, num_threads);
    for (ts_u32 i = 0; i < num_threads; i++) {
        tp->threads[i] = CreateThread(NULL, 0, _thread_start, tp, 0, NULL);

        if (tp->threads[i] == NULL) {
            TS_ERR(TS_ERR_THREADING, "Failed to create thread in thread pool");

            for (ts_u32 j = 0; j < i; j++) {
                TerminateThread(tp->threads[j], 0);
                CloseHandle(tp->threads[j]);
            }

            DeleteCriticalSection(&tp->mutex);

            return NULL;
        }
    }

    return tp;
}
void ts_thread_pool_destroy(ts_thread_pool* tp) {
    for (ts_u32 i = 0; i < tp->num_threads; i++) {
        // TODO: is it okay to use TerminateThread here?
        TerminateThread(tp->threads[i], 0);
        CloseHandle(tp->threads[i]);
    }

    DeleteCriticalSection(&tp->mutex);
}

ts_b32 ts_thread_pool_add_task(ts_thread_pool* tp, ts_thread_task task) {
    EnterCriticalSection(&tp->mutex);

    if ((ts_u64)tp->num_tasks + 1 >= (ts_u64)tp->max_tasks) {
        LeaveCriticalSection(&tp->mutex);

        TS_ERR(TS_ERR_THREADING, "Thread pool exceeded max tasks");
        return false;
    }

    tp->task_queue[tp->num_tasks++] = task;

    LeaveCriticalSection(&tp->mutex);
    WakeConditionVariable(&tp->queue_cond_var);

    return true;
}
ts_b32 ts_thread_pool_wait(ts_thread_pool* tp) {
    EnterCriticalSection(&tp->mutex);

    while (true) {
        if (tp->num_active != 0 || tp->num_tasks != 0) {
            SleepConditionVariableCS(&tp->active_cond_var, &tp->mutex, INFINITE);
        } else {
            break;
        }
    }

    LeaveCriticalSection(&tp->mutex);

    return true;
}

#endif // PLATFORM_WIN32

