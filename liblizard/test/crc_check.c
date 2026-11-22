// ai: ob_crc64, the block's BCH-derived CRC-64 (src/layout.c: carry-less multiplication where the processor has it,
// ai: PCLMULQDQ or PMULL, else sliced by eight through src/crc64_tables.h) against the bit loop: 200,000 random buffers
// ai: of 0 to 600 bytes and 20,000 of 600 to 4,000 at random offsets (every tail short of 16), every value compared, and
// ai: the standard check value; then a 472-byte block's time each way. Then ob_crc5, the format word's CRC-5 of its 35-bit
// ai: index (carry-less multiplication where the processor has it, Barrett's reduction), against its bit loop: every
// ai: 512th index and the first and last 65,536 (`crc_check`), or every one of the 2^35 (`crc_check crc5`, built with
// ai: -fopenmp: about 20 s on 32 threads), and a call's time each way. Exit 1 on any difference or a wrong check value.
// ai: Any platform; test/neon_kernels.sh runs it on a phone; on x86-64, from liblizard/ (-DOB_SCALAR the table path):
// ai: gcc -O3 -ffp-contract=off -DNDEBUG -msse4.1 -Wall -Wextra -Isrc -o build/crc_check test/crc_check.c
// ai: src/layout.c src/fmt.c src/rs.c src/ldpc.c && build/crc_check
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
uint64_t ob_crc64(const uint8_t *p, int n);
uint32_t ob_crc5(uint64_t c);
uint32_t ob_crc5_bits(uint64_t c);
static uint64_t loop(const uint8_t *p, int n) {
  uint64_t c = ~0ull;
  for (int i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xc02c38b47d32d307ull & (0ull - (c & 1))); }
  return ~c;
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
int main(int argc, char **argv) {
  const int every = argc > 1 && !strcmp(argv[1], "crc5");
  static uint8_t buf[4096 + 32]; uint32_t z = 5; int bad = 0;
  for (int t = 0; t < 220000; t++) {
    const int n = t < 200000 ? (int)(z % 601) : 600 + (int)(z % 3401), o = (int)((z >> 10) % 16);
    for (int i = 0; i < n + 16; i++) { z = z * 1664525u + 1013904223u; buf[i] = (uint8_t)(z >> 24); }
    if (ob_crc64(buf + o, n) != loop(buf + o, n)) bad++;
  }
  const uint64_t check = ob_crc64((const uint8_t *)"123456789", 9);
  if (check != 0xbb3ba0d234061b3eull) bad++;
  printf("CRC-64: 220000 buffers, %d apart (\"123456789\": %016llx, the check value bb3ba0d234061b3e)\n", bad,
         (unsigned long long)check);
  double ta = 1e30, tb = 1e30; volatile uint64_t sink = 0;
  for (int r = 0; r < 50; r++) {
    double t0 = now_ns(); for (int k = 0; k < 100; k++) sink += loop(buf, 472); double t = now_ns() - t0; if (t < ta) ta = t;
    t0 = now_ns(); for (int k = 0; k < 100; k++) sink += ob_crc64(buf, 472); t = now_ns() - t0; if (t < tb) tb = t;
  }
  printf("a 472-byte block: the bit loop %.0f ns, ob_crc64 %.0f ns (%.0fx)\n", ta / 100, tb / 100, ta / tb);
  // ai: the word's CRC-5 over its 35-bit indices
  const long long top = 1ll << 35, step = every ? 1 : 512;
  long long apart = 0, n5 = 0;
#pragma omp parallel for reduction(+ : apart, n5) schedule(static)
  for (long long c = 0; c < top; c += step) { apart += ob_crc5((uint64_t)c) != ob_crc5_bits((uint64_t)c); n5++; }
  for (long long c = 0; !every && c < 65536; c++) {
    apart += ob_crc5((uint64_t)c) != ob_crc5_bits((uint64_t)c) || ob_crc5((uint64_t)(top - 1 - c)) != ob_crc5_bits((uint64_t)(top - 1 - c));
    n5 += 2;
  }
  // ai: the C's own vector (SPEC 5.2): LIZARD-64's word at 60/1 is 4c 00 00 db c1 83, its index 450,060, its CRC-5 3
  if (ob_crc5(450060) != 3 || ob_crc5_bits(450060) != 3) apart++;
  bad += apart != 0;
  printf("CRC-5: %lld indices, %lld apart\n", n5, apart);
  double tc = 1e30, td = 1e30; volatile uint32_t s5 = 0;
  for (int r = 0; r < 50; r++) {
    double t0 = now_ns(); for (int k = 0; k < 1000; k++) s5 += ob_crc5_bits((uint64_t)k * 2654435761u); double t = now_ns() - t0; if (t < tc) tc = t;
    t0 = now_ns(); for (int k = 0; k < 1000; k++) s5 += ob_crc5((uint64_t)k * 2654435761u); t = now_ns() - t0; if (t < td) td = t;
  }
  printf("a word's CRC-5: the bit loop %.1f ns, ob_crc5 %.1f ns (%.1fx)\n", tc / 1000, td / 1000, tc / td);
  return bad != 0;
}
