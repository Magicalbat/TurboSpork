/**
 * @file os.h
 * @brief Operating system specific functions
 */

#ifndef OS_H
#define OS_H

#include "base_defs.h"
#include "str.h"
#include "mg/mg_arena.h"

/// File flags
typedef enum {
    /// File is directory
    TS_FILE_IS_DIR = (1 << 0)
} ts_file_flags;

// TODO: make this consistent between linux and windows
// TODO: confirm bounds here

/// Stores date and time
typedef struct {
    /**
     * @brief Seconds [0, 60]
     *
     * 60 is included because of leap seconds
     */
    ts_u8 sec;
    /// Minutes [0, 59]
    ts_u8 min;
    /// Hour [0, 23]
    ts_u8 hour;
    /// Day [1, 31]
    ts_u8 day;
    /// Month [1, 12]
    ts_u8 month;
    /// Year
    ts_i32 year;
} ts_datetime;

/// File stats
typedef struct {
    /// Size of file in bytes
    ts_u64 size;
    /// File flags
    ts_file_flags flags;
    /// Last time of modification
    ts_datetime modify_time;
} ts_file_stats;

/// Thread mutex
typedef struct _ts_mutex ts_mutex;

/// Thread pool
typedef struct _ts_thread_pool ts_thread_pool;

/// Function for thread to run
typedef void (ts_thread_func)(void*);
/**
 * @brief Task for thread to run
 *
 * Function will do `func(arg)`
 */
typedef struct {
    /// Function to run
    ts_thread_func* func;
    /// Function arg
    void* arg;
} ts_thread_task;

/**
 * @brief Initialize time system
 *
 * This is because of the windows API, see `os_windows.c` for more
 */
void ts_time_init(void);

/// Returns the local time
ts_datetime ts_now_localtime(void);
/// Returns time in microseconds
ts_u64 ts_now_usec(void);
/// Sleeps for `t` milliseconds
void ts_sleep_msec(ts_u32 t);

/**
 * @brief Reads entire file and returns it as a string8
 */
ts_string8 ts_file_read(mg_arena* arena, ts_string8 path);
/**
 * @brief Writes all strings in list to file
 *
 * @return true if write was successful, false otherwise
 */
ts_b32 ts_file_write(ts_string8 path, ts_string8_list str_list);
/**
 * @brief Gets stats of file
 */
ts_file_stats ts_file_get_stats(ts_string8 path);

/**
 * @brief Retrieves entropy from the OS
 *
 * @param data Where entropy is written to
 * @param size Number of bytes to retrieve
 */
void ts_get_entropy(void* data, ts_u64 size);

/**
 * @brief Creates a `ts_mutex`
 */
ts_mutex* ts_mutex_create(mg_arena* arena);
/// Destroys the mutex
void ts_mutex_destroy(ts_mutex* mutex);
/// Locks the mutex
void ts_mutex_lock(ts_mutex* mutex);
/// Unlocks the mutex
void ts_mutex_unlock(ts_mutex* mutex);

/**
 * @brief Creates a `ts_thread_pool`
 *
 * @param arena Arena to allocate thread pool on
 * @param num_threads Number of threads in pool.
 *  It should not be much higher than the number of threads on your computer
 * @param max_tasks Maximum number of tasks that can be active at one time
 */
ts_thread_pool* ts_thread_pool_create(mg_arena* arena, ts_u32 num_threads, ts_u32 max_tasks);
/// Destroys the thread pool
void ts_thread_pool_destroy(ts_thread_pool* tp);
/**
 * @brief Adds task to the thread pool's task queue
 */
void ts_thread_pool_add_task(ts_thread_pool* tp, ts_thread_task task);
/**
 * @brief Waits until all thread tasks are finished
 */
void ts_thread_pool_wait(ts_thread_pool* tp);

#endif // OS_H

