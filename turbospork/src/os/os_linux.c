#include "os.h"
#include "err.h"
#include "prng.h"

#ifdef TS_PLATFORM_LINUX

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>

void ts_time_init(void) { }

ts_datetime _tm_to_datetime(struct tm tm) {
    return (ts_datetime){
        .sec = tm.tm_sec,
        .min = tm.tm_min,
        .hour = tm.tm_hour,
        .day = tm.tm_mday,
        .month = tm.tm_mon + 1,
        .year = tm.tm_year + 1900
    };
}

ts_datetime ts_now_localtime(void) {
    time_t t = time(NULL);
    struct tm* tm_ptr = localtime(&t);

    if (tm_ptr == NULL) {
        TS_ERR(TS_ERR_OS, "Failed to convert time to localtime");

        return (ts_datetime){ 0 };
    }

    struct tm tm = *tm_ptr;

    return _tm_to_datetime(tm);
}
ts_u64 ts_now_usec(void) {
    struct timespec ts = { 0 };
    if (-1 == clock_gettime(CLOCK_MONOTONIC, &ts)) {
        TS_ERR(TS_ERR_OS, "Failed to get current time");
        return 0;
    }

    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
void ts_sleep_msec(ts_u32 t) {
    usleep(t * 1000);
}

int _open_impl(ts_string8 path, int flags, mode_t mode) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    ts_u8* path_cstr = ts_str8_to_cstr(scratch.arena, path);
    int fd = open((char*)path_cstr, flags, mode);

    mga_scratch_release(scratch);

    return fd;
}

ts_string8 ts_file_read(mg_arena* arena, ts_string8 path) {
    int fd = _open_impl(path, O_RDONLY, 0);
    
    if (fd == -1) {
        TS_ERR(TS_ERR_IO, "Failed to open file for reading");

        return (ts_string8){ 0 };
    }
    
    struct stat file_stat = { 0 };
    // This should not really error if the file was opened correctly
    fstat(fd, &file_stat);

    ts_string8 out = { 0 };

    if (S_ISREG(file_stat.st_mode)) {
        out.size = file_stat.st_size;
        out.str = MGA_PUSH_ZERO_ARRAY(arena, ts_u8, (ts_u64)file_stat.st_size);

        if (read(fd, out.str, file_stat.st_size) == -1) {
            TS_ERR(TS_ERR_IO, "Failed to read file");
            
            close(fd);
            
            return (ts_string8){ 0 };
        }
    } else {
        TS_ERR(TS_ERR_IO, "Failed to read file, file is not regular (most likely a directory)");
    }

    if (-1 == close(fd)) {
        TS_ERR(TS_ERR_IO, "Failed to close file");
    }

    return out;
}

ts_b32 ts_file_write(ts_string8 path, ts_string8_list str_list) {
    int fd = _open_impl(path, O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);

    if (fd == -1) {
        TS_ERR(TS_ERR_IO, "Failed to open file for writing");

        return false;
    }

    ts_b32 out = true;
    
    for (ts_string8_node* node = str_list.first; node != NULL; node = node->next) {
        ssize_t written = write(fd, node->str.str, node->str.size);

        if (written == -1) {
            TS_ERR(TS_ERR_IO, "Failed to write to file");

            out = false;
            break;
        }
    }
        
    if (-1 == close(fd)) {
        TS_ERR(TS_ERR_IO, "Failed to close file");
    }

    return out;
}
ts_file_flags _file_flags(mode_t mode) {
    ts_file_flags flags = { 0 };

    if (S_ISDIR(mode)) {
        flags |= TS_FILE_IS_DIR;
    }

    return flags;
}
ts_file_stats ts_file_get_stats(ts_string8 path) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    ts_u8* path_cstr = ts_str8_to_cstr(scratch.arena, path);
    
    struct stat file_stat = { 0 };
    
    int ret = stat((char*)path_cstr, &file_stat);
    
    mga_scratch_release(scratch);
    
    if (ret == -1) {
        TS_ERR(TS_ERR_IO, "Failed to get stats for file");

        return (ts_file_stats){ 0 };
    } 

    ts_file_stats stats = { 0 };
    
    time_t modify_time = (time_t)file_stat.st_mtim.tv_sec;
    struct tm* tm_ptr = localtime(&modify_time);
    struct tm tm = { 0 };

    // This error is recoverable
    if (tm_ptr == NULL) {
        TS_ERR(TS_ERR_OS, "Failed to convert time_t to localtime");
    } else {
        tm = *tm_ptr;
    }

    stats.size = file_stat.st_size;
    stats.flags = _file_flags(file_stat.st_mode);
    stats.modify_time = _tm_to_datetime(tm);

    return stats;
}

void ts_get_entropy(void* data, ts_u64 size) {
    if (-1 == getentropy(data, size)) {
        TS_ERR(TS_ERR_OS, "Failed to get entropy from system");
    }
}

typedef struct _ts_mutex {
    pthread_mutex_t mutex;
} ts_mutex;

// TODO: error handling here
ts_mutex* ts_mutex_create(mg_arena* arena) {
    pthread_mutex_t pmutex = PTHREAD_MUTEX_INITIALIZER;

    if (0 != pthread_mutex_init(&pmutex, NULL)) {
        TS_ERR(TS_ERR_THREADING, "Failed to create mutex");

        return NULL;
    }

    ts_mutex* mutex = MGA_PUSH_ZERO_STRUCT(arena, ts_mutex);
    mutex->mutex = pmutex;

    return mutex;
}
void ts_mutex_destroy(ts_mutex* mutex) {
    if (0 != pthread_mutex_destroy(&mutex->mutex)) {
        // Is there something else to do here?
        TS_ERR(TS_ERR_THREADING, "Failed to delete mutex");
    }
}
ts_b32 ts_mutex_lock(ts_mutex* mutex) {
    if (0 != pthread_mutex_lock(&mutex->mutex)) {
        TS_ERR(TS_ERR_THREADING, "Failed to lock mutex");

        return false;
    }
    return true;
}
ts_b32 ts_mutex_unlock(ts_mutex* mutex) {
    if (0 != pthread_mutex_unlock(&mutex->mutex)) {
        TS_ERR(TS_ERR_THREADING, "Failed to unlock mutex");

        return false;
    }

    return true;
}

typedef struct _ts_thread_pool {
    ts_u32 num_threads;
    pthread_t* threads;
    
    ts_b32 stop;

    ts_u32 max_tasks;
    ts_u32 num_tasks;
    ts_thread_task* task_queue;

    pthread_mutex_t mutex;
    pthread_cond_t queue_cond_var;

    ts_u32 num_active;
    pthread_cond_t active_cond_var;
} ts_thread_pool;

static void* linux_thread_start(void* arg) {
    ts_thread_pool* tp = (ts_thread_pool*)arg;
    ts_thread_task task = { 0 };

    // Init prng
    ts_u64 seeds[2] = { 0 };
    ts_get_entropy(seeds, sizeof(seeds));
    ts_prng_seed(seeds[0], seeds[1]);

    while (true) {
        pthread_mutex_lock(&tp->mutex);

        while (tp->num_tasks == 0 && !tp->stop) {
            pthread_cond_wait(&tp->queue_cond_var, &tp->mutex);
        }

        if (tp->stop) {
            break;
        }

        tp->num_active++;
        task = tp->task_queue[0];
        for (ts_u32 i = 0; i < tp->num_tasks - 1; i++) {
            tp->task_queue[i] = tp->task_queue[i + 1];
        }
        tp->num_tasks--;

        pthread_mutex_unlock(&tp->mutex);

        task.func(task.arg);

        pthread_mutex_lock(&tp->mutex);

        tp->num_active--;
        if (tp->num_active == 0) {
            pthread_cond_signal(&tp->active_cond_var);
        }

        pthread_mutex_unlock(&tp->mutex);
    }

    tp->num_threads--;
    pthread_cond_signal(&tp->active_cond_var);
    pthread_mutex_unlock(&tp->mutex);

    return NULL;
}

ts_thread_pool* ts_thread_pool_create(mg_arena* arena, ts_u32 num_threads, ts_u32 max_tasks) {
    mga_temp maybe_temp = mga_temp_begin(arena);

    ts_thread_pool* tp = MGA_PUSH_ZERO_STRUCT(arena, ts_thread_pool);

    tp->max_tasks = TS_MAX(num_threads, max_tasks);
    tp->task_queue = MGA_PUSH_ZERO_ARRAY(arena, ts_thread_task, max_tasks);

    ts_i32 ret = 0;
    ret |= pthread_mutex_init(&tp->mutex, NULL);
    ret |= pthread_cond_init(&tp->queue_cond_var, NULL);
    ret |= pthread_cond_init(&tp->active_cond_var, NULL);

    if (ret != 0) {
        TS_ERR(TS_ERR_THREADING, "Failed to init thread pool mutex and/or cond vars");

        mga_temp_end(maybe_temp);

        return NULL;
    }

    tp->num_threads = num_threads;
    tp->threads = MGA_PUSH_ZERO_ARRAY(arena, pthread_t, num_threads);
    for (ts_u32 i = 0; i < num_threads; i++) {
        ret = 0;

        ret |= pthread_create(&tp->threads[i], NULL, linux_thread_start, tp);
        ret |= pthread_detach(tp->threads[i]);

        if (ret != 0) {
            TS_ERR(TS_ERR_THREADING, "Failed to create or detach threads in thread pool");

            for (ts_u32 j = 0; j < i; j++) {
                pthread_cancel(tp->threads[j]);
            }

            pthread_cond_destroy(&tp->active_cond_var);
            pthread_mutex_destroy(&tp->mutex);
            pthread_cond_destroy(&tp->queue_cond_var);

            mga_temp_end(maybe_temp);

            return NULL;
        }
    }

    return tp;
}
void ts_thread_pool_destroy(ts_thread_pool* tp) {
    if (tp == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Unable to destroy NULL thread pool");

        return;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        TS_ERR(TS_ERR_THREADING, "Unable to destroy threadpool: cannot lock mutex");

        return;
    }

    tp->num_tasks = 0;

    tp->stop = true;
    pthread_cond_broadcast(&tp->queue_cond_var);

    pthread_mutex_unlock(&tp->mutex);

    ts_thread_pool_wait(tp);

    // Everything here should destroy correctly if the tp was initialized correctly 

    for (ts_u32 i = 0; i < tp->num_threads; i++) {
        pthread_cancel(tp->threads[i]);
    }

    pthread_mutex_destroy(&tp->mutex);
    pthread_cond_destroy(&tp->queue_cond_var);
    pthread_cond_destroy(&tp->active_cond_var);
}

ts_b32 ts_thread_pool_add_task(ts_thread_pool* tp, ts_thread_task task) {
    if (tp == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Unable to add task to NULL thread pool");

        return false;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        TS_ERR(TS_ERR_THREADING, "Failed to lock thread pool mutex, unable to add task");

        return false;
    }

    if ((ts_u64)tp->num_tasks + 1 > (ts_u64)tp->max_tasks) {
        pthread_mutex_unlock(&tp->mutex);

        TS_ERR(TS_ERR_THREADING, "Thread pool exceeded max tasks");

        return false;
    }

    tp->task_queue[tp->num_tasks++] = task;

    pthread_mutex_unlock(&tp->mutex);

    pthread_cond_signal(&tp->queue_cond_var);

    return true;
}
ts_b32 ts_thread_pool_wait(ts_thread_pool* tp) {
    if (tp == NULL) {
        TS_ERR(TS_ERR_INVALID_INPUT, "Unable to wait for NULL thread pool");

        return false;
    }

    if (0 != pthread_mutex_lock(&tp->mutex)) {
        TS_ERR(TS_ERR_THREADING, "Cannot wait for thread pool: unable to lock mutex");

        return false;
    }

    while (true) {
        //if (tp->num_active != 0 || tp->num_tasks != 0) {
        if ((!tp->stop && (tp->num_active != 0 || tp->num_tasks != 0)) || (tp->stop && tp->num_threads != 0)) {
            pthread_cond_wait(&tp->active_cond_var, &tp->mutex);
        } else {
            break;
        }
    }

    pthread_mutex_unlock(&tp->mutex);

    return true;
}

#endif

