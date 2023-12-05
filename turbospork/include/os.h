#ifndef OS_H
#define OS_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"

typedef enum {
    TS_FILE_IS_DIR = (1 << 0)
} ts_file_flags;

typedef struct {
    ts_u8 sec;
    ts_u8 min;
    ts_u8 hour;
    ts_u8 day;
    ts_u8 month;
    ts_i32 year;
} ts_datetime;

typedef struct {
    ts_u64 size;
    ts_file_flags flags;
    ts_datetime modify_time;
} ts_file_stats;

typedef struct _ts_mutex ts_mutex;

typedef struct _ts_thread_pool ts_thread_pool;

typedef void (ts_thread_func)(void*);
typedef struct {
    ts_thread_func* func;
    void* arg;
} ts_thread_task;

void ts_time_init(void);

ts_datetime ts_now_localtime(void);
ts_u64 ts_now_usec(void);
void ts_sleep_msec(ts_u32 t);

ts_string8 ts_file_read(mg_arena* arena, ts_string8 path);
ts_b32 ts_file_write(ts_string8 path, ts_string8_list str_list);
ts_file_stats ts_file_get_stats(ts_string8 path);
void ts_get_entropy(void* data, ts_u64 size);

ts_mutex* ts_mutex_create(mg_arena* arena);
void ts_mutex_destroy(ts_mutex* mutex);
void ts_mutex_lock(ts_mutex* mutex);
void ts_mutex_unlock(ts_mutex* mutex);

ts_thread_pool* ts_thread_pool_create(mg_arena* arena, ts_u32 num_threads, ts_u32 max_tasks);
void ts_thread_pool_destroy(ts_thread_pool* tp);
void ts_thread_pool_add_task(ts_thread_pool* tp, ts_thread_task task);
void ts_thread_pool_wait(ts_thread_pool* tp);

#endif // OS_H
