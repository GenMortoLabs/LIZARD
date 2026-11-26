// ai: jni_lib.h: dev.lizard.lib.LibNative's static methods over lizard.h's format numbers and the senders' bundle and
// ai: its reader (api/format.cpp, api/tar.cpp), registered by registerLib from the app's library's JNI_OnLoad. Strings
// ai: are true UTF-8 and UTF-16 (jni_text.h). A refusal throws: the bundle's IllegalArgumentException, the reader's
// ai: java.io.IOException, each with the C's sentence.
#include "jni_lib.h"

#include <cmath>
#include <string>
#include <vector>

#include "jni_text.h"
#include "lizard.h"

namespace lizard::jni {

namespace {

void raise(JNIEnv* e, const char* cls, const char* message) {
  if (jclass c = e->FindClass(cls)) e->ThrowNew(c, message);
}

// ai: [STEP, MIN_BLOCKS, MAX_BLOCKS, PAYLOAD, QUIET, BLOCKS_DEFAULT, GAP_MODULES, RINGS, the default ring's index, each
// ai: ring's cells, each ring's modules, TAR_END]: the numbers liblizard/kotlin/Lib.kt's Ladder reads once
jintArray consts(JNIEnv* e, jclass) {
  std::vector<jint> v = {LIZ_STEP, LIZ_MIN_BLOCKS, LIZ_MAX_BLOCKS, LIZ_PAYLOAD, LIZ_QUIET, LIZ_BLOCKS_DEFAULT, LIZ_GAP_MODULES, LIZ_RINGS, -1};
  const int def = liz_ring_cells(LIZ_RING_DEFAULT);
  for (int r = 0; r < LIZ_RINGS; r++) {
    v.push_back(liz_ring_cells(r));
    if (v.back() == def) v[8] = r;
  }
  for (int r = 0; r < LIZ_RINGS; r++) v.push_back(liz_ring_modules(r));
  v.push_back(LIZ_TAR_END);
  jintArray a = e->NewIntArray(jsize(v.size()));
  if (a) e->SetIntArrayRegion(a, 0, jsize(v.size()), v.data());
  return a;
}

liz_bundle* bb(jlong h) { return reinterpret_cast<liz_bundle*>(h); }
liz_untar* uu(jlong h) { return reinterpret_cast<liz_untar*>(h); }

// ai: the choice's items (paths, folders, sizes, times in seconds, NaN none) made one payload at now: the handle, info
// ai: [size, single, files, entries]; 0 with IllegalArgumentException where the C refuses it
jlong bundleMake(JNIEnv* e, jclass, jobjectArray paths, jbooleanArray dirs, jlongArray sizes, jdoubleArray mtimes, jdouble now, jlongArray info) {
  const jsize n = paths ? e->GetArrayLength(paths) : 0;
  if (!dirs || !sizes || !mtimes || !info || e->GetArrayLength(dirs) != n || e->GetArrayLength(sizes) != n || e->GetArrayLength(mtimes) != n || e->GetArrayLength(info) < 4) {
    raise(e, "java/lang/IllegalArgumentException", "a bundle's items: as many folders, sizes and times as paths, and four places for its info");
    return 0;
  }
  std::vector<jboolean> d(size_t(n) + 1);
  std::vector<jlong> s(size_t(n) + 1);
  std::vector<jdouble> t(size_t(n) + 1);
  e->GetBooleanArrayRegion(dirs, 0, n, d.data());
  e->GetLongArrayRegion(sizes, 0, n, s.data());
  e->GetDoubleArrayRegion(mtimes, 0, n, t.data());
  liz_bundle* b = liz_bundle_new();
  if (!b) { raise(e, "java/lang/OutOfMemoryError", liz_last_error()); return 0; }
  for (jsize i = 0; i < n; i++) {
    jstring p = static_cast<jstring>(e->GetObjectArrayElement(paths, i));
    const std::string path = str(e, p);
    e->DeleteLocalRef(p);
    if (liz_bundle_add(b, path.c_str(), d[size_t(i)] ? 1 : 0, s[size_t(i)], t[size_t(i)]) < 0) {
      raise(e, "java/lang/IllegalArgumentException", liz_last_error());
      liz_bundle_free(b);
      return 0;
    }
  }
  liz_bundle_info in{};
  if (liz_bundle_make(b, now, &in) < 0) {
    raise(e, "java/lang/IllegalArgumentException", liz_last_error());
    liz_bundle_free(b);
    return 0;
  }
  const jlong out[4] = {in.size, in.single, in.files, in.entries};
  e->SetLongArrayRegion(info, 0, 4, out);
  return reinterpret_cast<jlong>(b);
}

jstring bundleName(JNIEnv* e, jclass, jlong h) { return jstr(e, liz_bundle_name(bb(h))); }

// ai: entry k's head, out [size, item, pad]
jbyteArray bundleEntry(JNIEnv* e, jclass, jlong h, jint k, jlongArray out) {
  liz_bundle_entry x{};
  const uint8_t* head = liz_bundle_entry_get(bb(h), k, &x);
  if (!head || !out || e->GetArrayLength(out) < 3) { raise(e, "java/lang/IllegalArgumentException", liz_last_error()); return nullptr; }
  const jlong o[3] = {x.size, x.item, x.pad};
  e->SetLongArrayRegion(out, 0, 3, o);
  jbyteArray a = e->NewByteArray(x.head_len);
  if (a) e->SetByteArrayRegion(a, 0, x.head_len, reinterpret_cast<const jbyte*>(head));
  return a;
}

void bundleFree(JNIEnv*, jclass, jlong h) { liz_bundle_free(bb(h)); }

jlong untarNew(JNIEnv* e, jclass) {
  liz_untar* u = liz_untar_new();
  if (!u) raise(e, "java/lang/OutOfMemoryError", liz_last_error());
  return reinterpret_cast<jlong>(u);
}
void untarFree(JNIEnv*, jclass, jlong h) { liz_untar_free(uu(h)); }
jlong untarWant(JNIEnv*, jclass, jlong h) { return liz_untar_want(uu(h)); }

// ai: the first n bytes of b: LIZ_UNTAR_* (0 the end, 1 more, 2 an entry), or java.io.IOException with the C's sentence
jint untarTake(JNIEnv* e, jclass, jlong h, jbyteArray b, jint n) {
  if (n < 0 || (n > 0 && (!b || e->GetArrayLength(b) < n))) { raise(e, "java/lang/IllegalArgumentException", "untarTake: n bytes of the array"); return -1; }
  std::vector<uint8_t> bytes(size_t(n) + 1);
  if (n > 0) e->GetByteArrayRegion(b, 0, n, reinterpret_cast<jbyte*>(bytes.data()));
  const int r = liz_untar_take(uu(h), bytes.data(), size_t(n));
  if (r < 0) raise(e, "java/io/IOException", liz_last_error());
  return r;
}

// ai: the entry the last take named: its path, out [dir, size]
jstring untarEntry(JNIEnv* e, jclass, jlong h, jlongArray out) {
  int dir = 0;
  int64_t size = 0;
  const char* p = liz_untar_entry(uu(h), &dir, &size);
  if (out && e->GetArrayLength(out) >= 2) { const jlong o[2] = {dir, size}; e->SetLongArrayRegion(out, 0, 2, o); }
  return jstr(e, p);
}

jboolean loaded(JNIEnv*, jclass) { return JNI_TRUE; }

#define M(name, sig) {const_cast<char*>(#name), const_cast<char*>(sig), reinterpret_cast<void*>(name)}
#define S "Ljava/lang/String;"
const JNINativeMethod METHODS[] = {
    M(loaded, "()Z"),
    M(consts, "()[I"),
    M(bundleMake, "([" S "[Z[J[DD[J)J"),
    M(bundleName, "(J)" S),
    M(bundleEntry, "(JI[J)[B"),
    M(bundleFree, "(J)V"),
    M(untarNew, "()J"),
    M(untarFree, "(J)V"),
    M(untarWant, "(J)J"),
    M(untarTake, "(J[BI)I"),
    M(untarEntry, "(J[J)" S),
};
#undef S
#undef M

}  // namespace

bool registerLib(JNIEnv* e) {
  jclass c = e->FindClass("dev/lizard/lib/LibNative");
  if (!c) return false;
  return e->RegisterNatives(c, METHODS, sizeof METHODS / sizeof *METHODS) == JNI_OK;
}

}  // namespace lizard::jni
