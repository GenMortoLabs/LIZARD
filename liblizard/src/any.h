// ai: A receiver told nothing, as an object (lizard_phy.h lizard_acquire_ring; since 2026-09-27 the ring bootstraps what is
// ai: inside). A codec a ring, each at the smallest picture a format has (384, the top format there): the finder
// ai: registers against their layouts and the one that registered reads the word. The word names the sub-channels, so
// ai: the picture n; the frame is finished by the codec for that count in that ring, built the first time the pair
// ai: is seen and kept (LIZARD_ANY_PICS of them, the least recently used replaced), its layout the frame's word's
// ai: (lizard_phy.h lizard_layout). A frame whose word does not read is not finished: the word is read before the
// ai: frame (2026-11-18), and nothing carries from one frame to the next.
// ai: One lizard_any_t a thread: a decode keeps its registration inside the codecs it holds, so two threads never
// ai: share one, and the codec's per-thread state (internal.h OB_TLS) is then each thread's own. The wasm holds one
// ai: (wasm.c), behind its lizard_any_* exports.
#ifndef OB_ANY_H
#define OB_ANY_H
#include "lizard_phy.h"

enum { LIZARD_ANY_PICS = 2 };
typedef struct {
  lizard_t ring[LIZARD_RINGS], pic[LIZARD_ANY_PICS];
  // ai: a picture codec a (ring, sub-channel count): a frame decodes on its version's codec at its word's layout
  // ai: (lizard_phy.h lizard_layout), not as a prefix of its picture's top
  int ready, nmax, pic_ring[LIZARD_ANY_PICS], pic_n[LIZARD_ANY_PICS], pic_subch[LIZARD_ANY_PICS], pic_used[LIZARD_ANY_PICS], pic_clock;
  int which;               // the ring the last frame registered in, -1 none
  int n;                   // the picture it was finished at, 0 none
  int builds;              // pictures built since the init (a caller timing its first frames)
  const lizard_t *last;     // the codec that finished it
} lizard_any_t;

// ai: a: zeroed, or a receiver to set up again. nmax: the largest picture it decodes (1536, the ladder's top, in every
// ai: receiver since 2026-09-29; 1024 before). Returns the block bytes (472), or -1.
int lizard_any_init(lizard_any_t *a, int nmax);
void lizard_any_free(lizard_any_t *a);
// ai: One frame. blocks and ok hold lizard_any_top(a) blocks. Returns the blocks decoded; a->which and a->n say what
// ai: happened, res what the registration found.
int lizard_any_frame(lizard_any_t *a, const uint8_t *img, int iw, int ih, float gamma, uint8_t *blocks, uint8_t *ok, ob_result_t *res);
// ai: The most blocks a frame this receiver finishes can hold: the most any layout of any sub-channel count whose
// ai: picture fits nmax holds (lizard_phy.h lizard_blocks_max_for).
int lizard_any_top(const lizard_any_t *a);
// ai: The word the last frame's ring read, or 0.
const ob_fmt_t *lizard_any_word(const lizard_any_t *a);
// ai: The pilots the last frame's finish read (lizard_phy.h lizard_pilot) over the sub-channels its version carries: r of
// ai: the even sub-channels and of the odd, their standard errors, the sub-channels read; 0 when the frame was not finished.
int lizard_any_pilot(const lizard_any_t *a, float r[2], float sd[2]);
#endif
