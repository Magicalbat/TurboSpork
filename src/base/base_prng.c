#include "base_prng.h"

#include <math.h>

static THREAD_VAR prng s_rng = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL };

void prng_seed_r(prng* rng, u64 init_state, u64 init_seq) {
    rng->state = 0;
    rng->increment = (init_seq << 1) | 1;

    prng_rand_r(rng);

    rng->state += init_state;

    prng_rand_r(rng);
}
void prng_seed(u64 init_state, u64 init_seq) {
    prng_seed_r(&s_rng, init_state, init_seq);
}

u32 prng_rand_r(prng* rng) {
    u64 old_state = rng->state;

    rng->state = old_state * 6364136223846793005ULL + rng->increment;

    u32 xorshifted = ((old_state >> 18u) ^ old_state) >> 27u;
    u32 rot = old_state >> 59u;

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}
u32 prng_rand(void) {
    return prng_rand_r(&s_rng);
}

f32 prng_rand_f32_r(prng* rng) {
    return ldexpf(prng_rand_r(rng), -32);
}
f32 prng_rand_f32(void) {
    return prng_rand_f32_r(&s_rng);
}

