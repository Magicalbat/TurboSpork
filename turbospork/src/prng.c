#include "prng.h"

#include <math.h>

static TS_THREAD_VAR ts_prng s_rng = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL };

void ts_prng_seed_r(ts_prng* rng, ts_u64 init_state, ts_u64 init_seq) {
    rng->state = 0;
    rng->increment = (init_seq << 1) | 1;

    ts_prng_rand_r(rng);

    rng->state += init_state;

    ts_prng_rand_r(rng);
}
void ts_prng_seed(ts_u64 init_state, ts_u64 init_seq) {
    ts_prng_seed_r(&s_rng, init_state, init_seq);
}

ts_u32 ts_prng_rand_r(ts_prng* rng) {
    ts_u64 old_state = rng->state;

    rng->state = old_state * 6364136223846793005ULL + rng->increment;

    ts_u32 xorshifted = ((old_state >> 18u) ^ old_state) >> 27u;
    ts_u32 rot = old_state >> 59u;

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}
ts_u32 ts_prng_rand(void) {
    return ts_prng_rand_r(&s_rng);
}

ts_f32 ts_prng_rand_f32_r(ts_prng* rng) {
    return ldexpf(ts_prng_rand_r(rng), -32);
}
ts_f32 ts_prng_rand_f32(void) {
    return ts_prng_rand_f32_r(&s_rng);
}

// Box-Muller Transform
// https://en.wikipedia.org/wiki/Box–Muller_transform
ts_f32 ts_prng_std_norm_r(ts_prng* rng){
    static const ts_f32 epsilon = 1e-6;
    static const ts_f32 two_pi = 2.0 * 3.141592653f;

    ts_f32 u1 = epsilon;
    ts_f32 u2 = 0.0f;

    do {
        u1 = (ts_prng_rand_f32_r(rng)) * 2.0f - 1.0f;
    } while (u1 <= epsilon);
    u2 = (ts_prng_rand_f32_r(rng)) * 2.0f - 1.0f;

    ts_f32 mag = sqrtf(-2.0f * logf(u1));
    ts_f32 z0 = mag * cos(two_pi * u2);

    // I am ignoring the second value here
    // It might be worth trying to use it
    //ts_f32 z1 = mag * sin(two_pi * u2);

    return z0;
}

ts_f32 ts_prng_std_norm(void) {
    return ts_prng_std_norm_r(&s_rng);
}

