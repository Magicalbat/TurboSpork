
#include <inttypes.h>

#define TS_TRUE 1
#define TS_FALSE 0
#define TS_NULL (void*)0

typedef int8_t ts_i8;
typedef int16_t ts_i16;
typedef int32_t ts_i32;
typedef int64_t ts_i64;
typedef uint8_t ts_u8;
typedef uint16_t ts_u16;
typedef uint32_t ts_u32;
typedef uint64_t ts_u64;

typedef ts_i8 ts_b8;
typedef ts_i32 ts_b32;

typedef float ts_f32;
typedef double ts_f64;

#define TS_CONCAT_NX(a, b) a##b
#define TS_CONCAT(a, b) TS_CONCAT_NX(a, b)

#define TS_STATIC_ASSERT(c, id) static ts_u8 TS_CONCAT(id, __LINE__)[(c) ? 1 : -1]

TS_STATIC_ASSERT(sizeof(ts_f32) == 4, f32_size);
TS_STATIC_ASSERT(sizeof(ts_f64) == 8, f64_size);

#define TS_KiB(n) ((u64)(n) << 10)
#define TS_MiB(n) ((u64)(n) << 20)
#define TS_GiB(n) ((u64)(n) << 30)

#define TS_ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))
#define TS_ALIGN_DOWN_POW2(n, p) (((u64)(n)) & (~((u64)(p) - 1)))

#define TS_UNUSED(x) (void)(x)

#define TS_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define TS_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define TS_CLAMP(x, a, b) (MIN((b), MAX((x), (a))))
#define TS_ABS(n) ((n) < 0 ? -(n) : (n))
#define TS_SIGN(n) ((n) < 0 ? -1 : 1)

#define TS_SLL_PUSH_FRONT(f, l, n) ((f) == NULL ? \
    ((f) = (l) = (n)) :                        \
    ((n)->next = (f), (f) = (n)))              \

#define TS_SLL_PUSH_BACK(f, l, n) ((f) == NULL ? \
    ((f) = (l) = (n)) :                       \
    ((l)->next = (n), (l) = (n)),             \
    ((n)->next = NULL))                       \

#define TS_SLL_POP_FRONT(f, l) ((f) == (l) ? \
    ((f) = (l) = NULL) :                  \
    ((f) = (f)->next))                    \

