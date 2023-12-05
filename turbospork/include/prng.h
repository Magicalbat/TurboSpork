/** @file prng.h
 * @brief Pseudorandom number generator
 * 
 * Permuted congruential generator
 * Based on code from https://www.pcg-random.org
 */

#ifndef PRNG_H
#define PRNG_H

#include "base_defs.h"

/**
 * @brief Internal state for pseudorandom number generator.
 *
 * You can create your own `ts_prng` struct for a random number generator,
 * but there is also a static thread local random number generator.
 * Each function has a version that uses the static `ts_prng`
 */
typedef struct {
    ts_u64 state;
    ts_u64 increment;
} ts_prng;

/**
 * @brief Sets the random seed of the prng
 * 
 * Two ts_u64 numbers are requiered for the PCG algorithm 
 */ 
void ts_prng_seed_r(ts_prng* rng, ts_u64 init_state, ts_u64 init_seq);
/**
 * @brief Sets the random seed of the static thread local prng
 * 
 * See prng_seed_r for more detail
 */
void ts_prng_seed(ts_u64 init_state, ts_u64 init_seq);

/**
 * @brief Generates a pseudorandom ts_u32 given a ts_prng
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
