#ifndef OS_H
#define OS_H

#include "base/base.h"
#include "mg/mg_arena.h"

typedef enum {
    OS_FILE_IS_DIR = (1 << 0)
} os_file_flags;

typedef struct {
    u8 sec;
    u8 min;
    u8 hour;
    u8 day;
    u8 month;
    i32 year;
} os_datetime;

typedef struct {
    u64 size;
    os_file_flags flags;
    os_datetime modify_time;
} os_file_stats;

typedef struct _os_thread_mutex os_thread_mutex;

typedef struct _os_thread_pool os_thread_pool;

typedef void (os_thread_func)(void*);
typedef struct {
    os_thread_func* func;
    void* arg;
} os_thread_task;

void os_time_init(void);

os_datetime os_now_localtime(void);
u64 os_now_microseconds(void);
void os_sleep_milliseconds(u32 t);

string8 os_file_read(mg_arena* arena, string8 path);
b32 os_file_write(string8 path, string8_list str_list);
os_file_stats os_file_get_stats(string8 path);

os_thread_mutex* os_thread_mutex_create(mg_arena* arena);
void os_thread_mutex_destroy(os_thread_mutex* mutex);
void os_thread_mutex_lock(os_thread_mutex* mutex);
void os_thread_mutex_unlock(os_thread_mutex* mutex);

os_thread_pool* os_thread_pool_create(mg_arena* arena, u32 num_threads, u32 max_tasks);
void os_thread_pool_destroy(os_thread_pool* tp);
void os_thread_pool_add_task(os_thread_pool* tp, os_thread_task task);
void os_thread_pool_wait(os_thread_pool* tp);

#endif // OS_H
