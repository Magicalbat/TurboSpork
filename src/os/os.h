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

os_datetime os_now_localtime(void);
u64 os_now_microseconds(void);
void os_sleep_milliseconds(u32 t);

string8 os_file_read(mg_arena* arena, string8 path);
b32 os_file_write(string8 path, string8_list str_list);
b32 os_file_append(string8 path, string8_list str_list);
os_file_stats os_file_get_stats(string8 path);

#endif // OS_H