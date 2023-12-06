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
 * @brief Sets the random seed of the `ts_prng`
 * 
 * Two `ts_u64` numbers are requiered for the PCG algorithm 
 */ 
void ts_prng_seed_r(ts_prng* rng, ts_u64 init_state, ts_u64 init_seq);
/**
 * @brief Sets the random seed of the static thread local `ts_prng`
 * 
 * See `ts_prng_seed_r` for more detail
 */
void ts_prng_seed(ts_u64 init_state, ts_u64 init_seq);

/**
 * @brief Generates a pseudorandom `ts_u32` given a `ts_prng`
 * 
 * @param rng Random number generator state
 */
ts_u32 ts_prng_rand_r(ts_prng* rng);
/**
 * @brief Generates a pseudorandom `ts_u32` from the static thread local `ts_prng`
 */
ts_u32 ts_prng_rand(void);

/**
 * @brief Generates a pseudorandom `ts_f32` in the range [0, 1) given a `ts_prng`
 * 
 * @param rng Random number generator state
 */
ts_f32 ts_prng_rand_f32_r(ts_prng* rng);
/**
 * @brief Generates a pseudorandom `ts_f32` in the range [0, 1) given from the static thread local `ts_prng`
 */
ts_f32 ts_prng_rand_f32(void);

/**
 * @brief Generates a pseudorandom and normally distributed number given a `ts_prng`
 *
 * Uses the Box-Muller transform: https://en.wikipedia.org/wiki/Box–Muller_transform
 * 
 * @param rng Random number generator state
 */
ts_f32 ts_prng_std_norm_r(ts_prng* rng);
/**
 * @brief Generates a pseudorandom and normally distributed number from the static thread local `ts_prng`
 *
 * See `ts_prng_std_norm_r` for more detail
 */
ts_f32 ts_prng_std_norm(void);

#endif // PRNG_H
