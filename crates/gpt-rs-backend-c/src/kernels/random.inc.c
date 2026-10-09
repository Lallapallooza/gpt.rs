
/* One block of Philox4x32-10 (Random123): the four 32-bit output words for the 128-bit counter
 * `counter + j` and the 64-bit `key`. The counter and key split into 32-bit words, least
 * significant first. */
static inline void gpt_rs_philox4x32_10(uint64_t key, uint64_t counter, uint64_t j,
                                        uint32_t out[4]) {
    const uint64_t low = counter + j;
    const uint64_t high = low < counter ? 1u : 0u;
    uint32_t c[4] = {(uint32_t)low, (uint32_t)(low >> 32), (uint32_t)high, 0u};
    uint32_t k[2] = {(uint32_t)key, (uint32_t)(key >> 32)};
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k[0] += 0x9E3779B9u;
            k[1] += 0xBB67AE85u;
        }
        const uint64_t p0 = (uint64_t)0xD2511F53u * c[0];
        const uint64_t p1 = (uint64_t)0xCD9E8D57u * c[2];
        const uint32_t next[4] = {
            (uint32_t)(p1 >> 32) ^ c[1] ^ k[0],
            (uint32_t)p1,
            (uint32_t)(p0 >> 32) ^ c[3] ^ k[1],
            (uint32_t)p0,
        };
        c[0] = next[0];
        c[1] = next[1];
        c[2] = next[2];
        c[3] = next[3];
    }
    out[0] = c[0];
    out[1] = c[1];
    out[2] = c[2];
    out[3] = c[3];
}
