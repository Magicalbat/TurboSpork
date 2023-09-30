#include "os.h"

#ifdef PLATFORM_LINUX

#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

#define _lnx_error(msg, ...) \
    fprintf(stderr, msg ", Linux Error: %s", __VA_ARGS__, strerror(errno))

os_datetime _tm_to_datetime(struct tm tm) {
    return (os_datetime){
        .sec = tm.tm_sec,
        .min = tm.tm_min,
        .hour = tm.tm_hour,
        .day = tm.tm_mday,
        .month = tm.tm_mon + 1,
        .year = tm.tm_year + 1900
    };
}

os_datetime os_now_localtime(void) {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);

    return _tm_to_datetime(tm);
}
u64 os_now_microseconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}
void os_sleep_milliseconds(u32 t) {
    usleep(t * 1000);
}

int _open_impl(string8 path, int flags, mode_t mode) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    u8* path_cstr = str8_to_cstr(scratch.arena, path);
    int fd = open((char*)path_cstr, flags, mode);

    mga_scratch_release(scratch);

    return fd;
}

string8 os_file_read(mg_arena* arena, string8 path) {
    int fd = _open_impl(path, O_RDONLY, 0);
    
    if (fd == -1) {
        _lnx_error("Failed to open file \"%.*s\"", (int)path.size, path.str);

        return (string8){ 0 };
    }
    
    struct stat file_stat;
    fstat(fd, &file_stat);

    string8 out = { 0 };

    if (S_ISREG(file_stat.st_mode)) {
        out.size = file_stat.st_size;
        out.str = MGA_PUSH_ZERO_ARRAY(arena, u8, (u64)file_stat.st_size);

        if (read(fd, out.str, file_stat.st_size) == -1) {
            _lnx_error("Failed to read file \"%.*s\"", (int)path.size, path.str);
            
            close(fd);
            
            return (string8){ 0 };
        }
    } else {
        fprintf(stderr, "Failed to read file \"%.*s\", file is not regular", (int)path.size, path.str);
    }
    close(fd);

    return out;
}

b32 os_file_write(string8 path, string8_list str_list) {
    int fd = _open_impl(path, O_CREAT | O_TRUNC | O_WRONLY, S_IRUSR | S_IWUSR);

    if (fd == -1) {
        _lnx_error("Failed to open file \"%.*s\"", (int)path.size, path.str);

        return false;
    }

    b32 out = true;
    
    for (string8_node* node = str_list.first; node != NULL; node = node->next) {
        ssize_t written = write(fd, node->str.str, node->str.size);

        if (written == -1) {
            _lnx_error("Failed to write to file \"%.*s\"", (int)path.size, path.str);

            out = false;
            break;
        }
    }
        
    close(fd);

    return out;
}
b32 os_file_append(string8 path, string8_list str_list) {
    int fd = _open_impl(path, O_APPEND | O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);

    if (fd == -1) {
        _lnx_error("Failed to open file \"%.*s\"", (int)path.size, path.str);

        return false;
    }
    
    b32 out = true;

    for (string8_node* node = str_list.first; node != NULL; node = node->next) {
        ssize_t written = write(fd, node->str.str, node->str.size);
        
        if (written == -1) {
            _lnx_error("Failed to append to file \"%.*s\"", (int)path.size, path.str);

            out = false;
            break;
        }
    }

    close(fd);
    
    return out;
}
os_file_flags _file_flags(mode_t mode) {
    os_file_flags flags = { 0 };

    if (S_ISDIR(mode)) {
        flags |= OS_FILE_IS_DIR;
    }

    return flags;
}
os_file_stats os_file_get_stats(string8 path) {
    mga_temp scratch = mga_scratch_get(NULL, 0);
    
    u8* path_cstr = str8_to_cstr(scratch.arena, path);
    
    struct stat file_stat;
    
    int ret = stat((char*)path_cstr, &file_stat);
    
    mga_scratch_release(scratch);
    
    if (ret == -1) {
        _lnx_error("Failed to get stats for file \"%.*s\"", (int)path.size, path.str);

        return (os_file_stats){ 0 };
    } 

    os_file_stats stats = { 0 };
    
    time_t modify_time = (time_t)file_stat.st_mtim.tv_sec;
    struct tm tm = *localtime(&modify_time);

    stats.size = file_stat.st_size;
    stats.flags = _file_flags(file_stat.st_mode);
    stats.modify_time = _tm_to_datetime(tm);

    return stats;
}



#endif