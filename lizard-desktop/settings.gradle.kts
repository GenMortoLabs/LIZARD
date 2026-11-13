pluginManagement {
    repositories { gradlePluginPortal(); mavenCentral(); google() }
}
// ai: JDKs the build asks for and does not find installed (the packages' runtime, JDK 25: build.gradle.kts) downloaded
// ai: by Gradle's toolchain resolver (foojay), here and on CI
plugins { id("org.gradle.toolchains.foojay-resolver-convention") version "0.10.0" }
dependencyResolutionManagement {
    repositories { mavenCentral(); google() }
}
rootProject.name = "lizard-sender"
// ai: the apps' shared UI (Compose Multiplatform, ../lizard-ui), built as part of this build; the Android app includes it alike
include(":lizard-ui")
project(":lizard-ui").projectDir = file("../lizard-ui")
