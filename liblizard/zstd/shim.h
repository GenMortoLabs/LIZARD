// zstd for the default client's pack (core/tx/pack.h, sim/pack.mjs; header version 4, src/xfer.h: the transfer
// compresses nothing, a client packs a file before it goes), as every LIZARD build calls it: the parameters are pinned
// here, so the web's client in wasm and the native apps write the same frames for the same file (the same packed
// bytes, the same root: a transfer resumed from another client's send of that file goes on), and any receiver reads
// any client's. Changing a parameter changes what clients send, not what receivers can read: a zstd frame is a frame.
#ifndef LIZARD_ZSTD_SHIM_H
#define LIZARD_ZSTD_SHIM_H
#include <stdint.h>

// The compression level, the one choice. zstd's levels trade time for size along a knee: from 3 to 9 about 7% fewer
// bytes for five times the work, from 9 to 19 about 5% fewer for twelve times. A client packs the whole file before
// its first frame (the header carries the packed file's root), so 9 keeps a phone's start to seconds on a hundred
// megabytes where 19 would be a minute, and gives up a few percent of bytes against it.
#define LIZARD_ZSTD_LEVEL 9

#ifdef __cplusplus
extern "C" {
#endif
// The most bytes a frame of n bytes can take (ZSTD_compressBound).
uint32_t lizard_zstd_bound(uint32_t n);
// One frame of src[0..n) into dst (cap bytes): the frame's bytes, or -1 where zstd refuses (cap too small).
int32_t lizard_zstd_compress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t cap);
// The len bytes a frame of n bytes holds, into dst: 0, or -1 where the frame does not say it holds len bytes, is
// damaged, or decompresses to anything but len.
int32_t lizard_zstd_decompress(const uint8_t *src, uint32_t n, uint8_t *dst, uint32_t len);
// What a frame at src[0..n) says it holds (its header, 18 bytes at most, is enough): its bytes, or -1 where it says
// none, is not a frame, or holds 2^31 bytes or more.
int32_t lizard_zstd_frame_content(const uint8_t *src, uint32_t n);
// The bytes the frame at src takes, where src[0..n) holds it whole: or -1.
int32_t lizard_zstd_frame_bytes(const uint8_t *src, uint32_t n);
#ifdef __cplusplus
}
#endif
#endif
