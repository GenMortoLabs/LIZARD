// ai: The C API's lower layers as both apps reach them (jni_lib.cpp, 2026-11-25): dev.lizard.lib.LibNative's methods
// ai: registered on the class (RegisterNatives) from the JNI_OnLoad of the app's library (the Android app's libliz, the
// ai: desktop's lizard_desktop), as the session's are (jni_session.h): the format's numbers the apps show and paint by,
// ai: and the senders' bundle and its reader (lizard.h liz_bundle_*, liz_untar_*).
#pragma once
#include <jni.h>

namespace lizard::jni {

// ai: false where the class or a method is missing (the caller's JNI_OnLoad then fails the load)
bool registerLib(JNIEnv* e);

}  // namespace lizard::jni
