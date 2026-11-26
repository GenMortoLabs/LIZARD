// A listed frame's configuration from the gate's LISTS (core/dec/build.cpp backHalf: its first word at listsBlocks(B)
// + f, its 1/2 count at listsHalf(B) + f; lib/cfg.slang), B the batch's frames. Defines: SLOT_COUNT (uint).
#define CFG_AT(LISTS_, B_, F_) uint2(LISTS_[SLOT_COUNT * (B_ + 1u) + 2u * B_ + F_], LISTS_[SLOT_COUNT * (B_ + 1u) + 6u * B_ + F_])
