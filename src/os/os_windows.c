#include "os.h"

#ifdef PLATFORM_WIN32

#include <stdio.h>

#ifndef UNICODE
    #define UNICODE
#endif
#define WIN32_LEAN_AND_MEAN

#include <Windows.h>
#include <timeapi.h>

static u64 _ticks_per_sec = 1;

static string8 _error_string(mg_arena* arena) {
    DWORD err = GetLastError();
    if (err == 0) {
        return (string8){ 0 };
    }

    LPSTR msg_buf = NULL;
    DWORD msg_size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
        (LPSTR)&msg_buf, // Very intuitive win32
        0, NULL
    );

    string8 out;
    out.size = (u64)msg_size - 3;
    out.str = MGA_PUSH_ZERO_ARRAY(arena, u8, (u64)msg_size - 3);

    memcpy(out.str, msg_buf, msg_size);

    LocalFree(msg_buf);

    return out;
}

#define _w32_error(msg, ...) do {\
        mga_temp scratch = mga_scratch_get(NULL, 0); \
        string8 err_str = _error_string(scratch.arena); \
        fprintf(stderr, msg ", Win32 Error: %.*s\n", __VA_ARGS__, (int)err_str.size, (char*)err_str.str); \
        mga_scratch_release(scratch); \
    } while (0)

void os_time_init(void) {
    LARGE_INTEGER perf_freq;
    if (QueryPerformanceFrequency(&perf_freq)) {
        _ticks_per_sec = ((u64)perf_freq.HighPart << 32) | perf_freq.LowPart;
    }
}

static os_datetime _systime_to_datetime(SYSTEMTIME t){  
    return (os_datetime){
        .sec   = (u8 )t.wSecond,
        .min   = (u8 )t.wMinute,
        .hour  = (u8 )t.wHour,
        .day   = (u8 )t.wDay,
        .month = (u8 )t.wMonth,
        .year  = (i32)t.wYear
    };
}

os_datetime os_now_localtime(void) {
    SYSTEMTIME t;
    GetLocalTime(&t);
    return _systime_to_datetime(t);
}

u64 os_now_microseconds(void) {
    u64 out = 0;
    LARGE_INTEGER perf_count;
    if (QueryPerformanceCounter(&perf_count)) {
        u64 ticks = ((u64)perf_count.HighPart << 32) | perf_count.LowPart;
        out = ticks * 1000000 / _ticks_per_sec;
    }
    return out;
}
void os_sleep_milliseconds(u32 t) {
    Sleep(t);
}

string8 os_file_read(mg_arena* arena, string8 path) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string16 path16 = str16_from_str8(scratch.arena, path);

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
        _w32_error("Failed to open file \"%.*s\"", (int)path.size, (char*)path.str);

        return (string8){ 0 };
    }

    string8 out = { 0 };

    DWORD high_size = 0;
    DWORD low_size = GetFileSize(file_handle, &high_size);
    u64 total_size = ((u64)high_size << 32) | low_size;

    mga_temp possible_temp = mga_temp_begin(arena);

    u8* buffer = MGA_PUSH_ZERO_ARRAY(arena, u8, total_size);

    u64 total_read = 0;
    while (total_read < total_size) {
        u64 to_read64 = total_size - total_read;
        DWORD to_read = to_read64 > ~(DWORD)(0) ? ~(DWORD)(0) : (DWORD)to_read64;

        DWORD bytes_read = 0;
        if (ReadFile(file_handle, buffer + total_read, to_read, &bytes_read, 0) == FALSE) {
            _w32_error("Failed to read to file \"%.*s\"", (int)path.size, (char*)path.str);
            mga_temp_end(possible_temp);

            return (string8){ 0 };
        }

        total_read += bytes_read;
    }

    out.size = total_size;
    out.str = buffer;

    CloseHandle(file_handle);
    return out;
}

b32 _file_write_impl(HANDLE file_handle, string8_list str_list) {
    for (string8_node* node = str_list.first; node != NULL; node = node->next) {
        u64 total_to_write = node->str.size;
        u64 total_written = 0;

        while (total_written < total_to_write) {
            u64 to_write64 = total_to_write - total_written;
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

b32 os_file_write(string8 path, string8_list str_list) {
    mga_temp scratch = mga_scratch_get(NULL, 0);

    string16 path16 = str16_from_str8(scratch.arena, path);

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
        _w32_error("Failed to open file \"%.*s\"", (int)path.size, (char*)path.str);

        return false;
    }

    b32 out = true;

    if (!_file_write_impl(file_handle, str_list)) {
        _w32_error("Failed to write to file \"%.*s\"", (int)path.size, (char*)path.str);

        out = false;
    }

    CloseHandle(file_handle);

    return out;

}
os_file_stats os_file_get_stats(string8 path) {
    os_file_stats stats = { 0 };

    mga_temp scratch = mga_scratch_get(NULL, 0);

    string16 path16 = str16_from_str8(scratch.arena, path);

    WIN32_FILE_ATTRIBUTE_DATA attribs = { 0 };
    if (GetFileAttributesEx((LPCWSTR)path16.str, GetFileExInfoStandard, &attribs) == FALSE) {
        _w32_error("Failed to get stats for file \"%.*s\"",  (int)path.size, (char*)path.str);
    }

    mga_scratch_release(scratch);

    stats.size = ((u64)attribs.nFileSizeHigh << 32) | attribs.nFileSizeLow;

    if (attribs.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        stats.flags |= OS_FILE_IS_DIR;
    }

    SYSTEMTIME modify_sys_time = { 0 };
    FileTimeToSystemTime(&attribs.ftLastWriteTime, &modify_sys_time);
    stats.modify_time = _systime_to_datetime(modify_sys_time);

    return stats;
}

typedef struct _os_thread_mutex {
    CRITICAL_SECTION cs;
} os_thread_mutex;

os_thread_mutex* os_thread_mutex_create(mg_arena* arena) {
    os_thread_mutex* mutex = MGA_PUSH_ZERO_STRUCT(arena, os_thread_mutex);

    InitializeCriticalSection(&mutex->cs);

    return mutex;
}
void os_thread_mutex_destroy(os_thread_mutex* mutex) {
    DeleteCriticalSection(&mutex->cs);
}
void os_thread_mutex_lock(os_thread_mutex* mutex) {
    EnterCriticalSection(&mutex->cs);
}
void os_thread_mutex_unlock(os_thread_mutex* mutex) {
    LeaveCriticalSection(&mutex->cs);
}

typedef struct _os_thread_pool {
    u32 num_threads;
    HANDLE* threads;

    u32 max_tasks;
    u32 num_tasks;
    os_thread_task* task_queue;

    CRITICAL_SECTION mutex; // I know that it is not technically a mutex on win32
    CONDITION_VARIABLE queue_cond_var;

    u32 num_active;
    CONDITION_VARIABLE active_cond_var;
} os_thread_pool;

static DWORD _thread_start(void* arg) {
    os_thread_pool* tp = (os_thread_pool*)arg;
    os_thread_task task = { 0 };

    while (true) {
        EnterCriticalSection(&tp->mutex);
        while (tp->num_tasks == 0) {
            SleepConditionVariableCS(&tp->queue_cond_var, &tp->mutex, INFINITE);
        }

        tp->num_active++;
        task = tp->task_queue[0];
        for (u32 i = 0; i < tp->num_tasks - 1; i++) {
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

os_thread_pool* os_thread_pool_create(mg_arena* arena, u32 num_threads, u32 max_tasks) {
    os_thread_pool* tp = MGA_PUSH_ZERO_STRUCT(arena, os_thread_pool);

    tp->max_tasks = max_tasks;
    tp->task_queue = MGA_PUSH_ZERO_ARRAY(arena, os_thread_task, max_tasks);

    InitializeCriticalSection(&tp->mutex);
    InitializeConditionVariable(&tp->queue_cond_var);
    InitializeConditionVariable(&tp->active_cond_var);

    tp->num_threads = num_threads;
    tp->threads = MGA_PUSH_ZERO_ARRAY(arena, HANDLE, num_threads);
    for (u32 i = 0; i < num_threads; i++) {
        tp->threads[i] = CreateThread(
            NULL, 0, _thread_start, tp, 0, NULL
        );
    }

    return tp;
}
void os_thread_pool_destroy(os_thread_pool* tp) {
    for (u32 i = 0; i < tp->num_threads; i++) {
        CloseHandle(tp->threads[i]);
    }

    DeleteCriticalSection(&tp->mutex);
}

void os_thread_pool_add_task(os_thread_pool* tp, os_thread_task task) {
    EnterCriticalSection(&tp->mutex);

    if ((u64)tp->num_tasks + 1 >= (u64)tp->max_tasks) {
        LeaveCriticalSection(&tp->mutex);
        fprintf(stderr, "Thread pool exceeded max tasks\n");
        return;
    }

    tp->task_queue[tp->num_tasks++] = task;

    LeaveCriticalSection(&tp->mutex);
    WakeConditionVariable(&tp->queue_cond_var);
}
void os_thread_pool_wait(os_thread_pool* tp) {
    EnterCriticalSection(&tp->mutex);
    while (true) {
        if (tp->num_active != 0 || tp->num_tasks != 0) {
            SleepConditionVariableCS(&tp->active_cond_var, &tp->mutex, INFINITE);
        } else {
            break;
        }
    }
    LeaveCriticalSection(&tp->mutex);
}

#endif // PLATFORM_WIN32
