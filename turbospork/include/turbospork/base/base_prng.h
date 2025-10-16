// Permuted congruential generator
// Based on https://www.pcg-random.org

typedef struct {
    ts_u64 state;
    ts_u64 increment;
} ts_prng_context;

void ts_prng_seed_r(ts_prng_context* rng, ts_u64 init_state, ts_u64 init_seq);
void ts_prng_seed(ts_u64 init_state, ts_u64 init_seq);

ts_u32 ts_prng_rand_r(ts_prng_context* rng);
ts_u32 ts_prng_rand(void);

// Generates a random number between 0 and 1
ts_f32 ts_prng_randf_r(ts_prng_context* rng);
ts_f32 ts_prng_randf(void);

ts_f32 ts_prng_std_norm_r(ts_prng_context* rng);
ts_f32 ts_prng_std_norm(void);

