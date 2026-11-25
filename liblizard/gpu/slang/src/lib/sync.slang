// A value every lane of the workgroup reads after one lane wrote it to workgroup memory: a barrier, the read, a
// barrier (the next write waits for every read).
#define UNIFORM_LOAD(DECL, SRC) GroupMemoryBarrierWithGroupSync(); DECL = SRC; GroupMemoryBarrierWithGroupSync();
// The workgroup's storage-buffer writes visible to its every lane, with the barrier (device scope: RADV did not show a
// workgroup-scope storage barrier's writes to the other lanes, 2026-11-16)
#define STORAGE_BARRIER DeviceMemoryBarrierWithGroupSync();
