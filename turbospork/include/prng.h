#ifndef PRNG_H
#define PRNG_H

// Permuted congruential generator
// Based on code from https://www.pcg-random.org/

#include "base_defs.h"

/**
 * Internal state for pseudorandom number generator <br>
 * Uses the PCG algorithm <br>
 * Code from https://www.pcg-random.org 
 */
typedef struct {
    ts_u64 state;
    ts_u64 increment;
} ts_prng;

/**
 * Sets the random seed of the prng <br>
 * Two ts_u64 numbers are requiered for the PCG algorithm 
 */ 
void ts_prng_seed_r(ts_prng* rng, ts_u64 init_state, ts_u64 init_seq);
/**
 * Sets the random seed of the static thread local prng <br>
 * See prng_seed_r for more detail
 */
void ts_prng_seed(ts_u64 init_state, ts_u64 init_seq);

/**
 * Generates a pseudorandom ts_u32 
 * 
 * @param rng Random number generator state
 */
ts_u32 ts_prng_rand_r(ts_prng* rng);
ts_u32 ts_prng_rand(void);

ts_f32 ts_prng_rand_f32_r(ts_prng* rng);
ts_f32 ts_prng_rand_f32(void);

ts_f32 ts_prng_std_norm_r(ts_prng* rng);
ts_f32 ts_prng_std_norm(void);

#endif // PRNG_H
