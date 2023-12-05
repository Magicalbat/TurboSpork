#ifndef BASE_DEFS_H
#define BASE_DEFS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>

typedef int8_t   ts_i8;
typedef int16_t  ts_i16;
typedef int32_t  ts_i32;
typedef int64_t  ts_i64;
typedef uint8_t  ts_u8;
typedef uint16_t ts_u16;
typedef uint32_t ts_u32;
typedef uint64_t ts_u64;

typedef ts_i8  ts_b8;
typedef ts_i32 ts_b32;

typedef float  ts_f32;
typedef double ts_f64;

static_assert(sizeof(ts_f32) == 4, "f32 size");
static_assert(sizeof(ts_f64) == 8, "f64 size");

#if defined(_WIN32)
#   define TS_PLATFORM_WIN32
#elif defined(__linux__)
#   define TS_PLATFORM_LINUX
#endif

#ifndef TS_THREAD_VAR
#    if defined(__clang__) || defined(__GNUC__)
#        define TS_THREAD_VAR __thread
#    elif defined(_MSC_VER)
#        define TS_THREAD_VAR __declspec(thread)
#    elif (__STDC_VERSION__ >= 201112L)
#        define TS_THREAD_VAR _Thread_local
#    else
#        error "Invalid compiler/version for thread var"
#    endif
#endif



#define TS_UNUSED(x) (void)(x)

#define TS_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define TS_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define TS_ABS(n) ((n) < 0 ? -(n) : (n))
#define TS_SIGN(n) ((n) < 0 ? -1 : 1)

#define TS_SLL_PUSH_FRONT(f, l, n) ((f) == 0 ? \
    ((f) = (l) = (n)) :                     \
    ((n)->next = (f), (f) = (n)))           \

#define TS_SLL_PUSH_BACK(f, l, n) ((f) == 0 ? \
    ((f) = (l) = (n)) :                    \
    ((l)->next = (n), (l) = (n)),          \
    ((n)->next = 0))                       \

#define TS_SLL_POP_FRONT(f, l) ((f) == (l) ? \
    ((f) = (l) = 0) :                     \
    ((f) = (f)->next))                    \

#endif // BASE_DEFS_H

