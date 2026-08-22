plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.compose.compiler)
    id("com.vanniktech.maven.publish") version "0.37.0"
}

val tag: String = if (System.getenv("GITHUB_REF_TYPE") == "tag") {
    System.getenv("GITHUB_REF_NAME")
} else {
    val baseVersion = providers.exec {
        commandLine("git", "rev-parse", "--short", "HEAD")
    }.standardOutput.asText.map { it.trim() }.getOrElse("unknown")
    "$baseVersion-SNAPSHOT"
}

android {
    namespace = "ca.mpreg.webgpuviewer"
    compileSdk = 37

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("proguard-rules.txt")

        externalNativeBuild {
            cmake {
                cppFlags("-O3 -flto")
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures {
        compose = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.annotation)
    implementation(libs.androidx.core)
    implementation(libs.androidx.compose.foundation)

    api(libs.androidx.webgpu)
}

afterEvaluate {
    mavenPublishing {
        coordinates("io.github.rianza", "webgpuviewer", tag)

        pom {
            name.set("webgpuviewer")
            description.set("WebGPU-based image viewer for Android (Adreno GLES fix fork)")
            inceptionYear.set("2026")
            url.set("https://github.com/rianza/webgpuviewer")
            licenses {
                license {
                    name.set("MIT License")
                    url.set("https://opensource.org")
                    distribution.set("repo")
                }
            }
            developers {
                developer {
                    id.set("rianza")
                    name.set("rianza")
                    url.set("https://github.com/rianza/")
                }
            }
            scm {
                url.set("https://github.com/rianza/webgpuviewer/")
                connection.set("scm:git:git://github.com/rianza/webgpuviewer.git")
                developerConnection.set("scm:git:ssh://github.com/rianza/webgpuviewer.git")
            }
        }

        publishToMavenCentral(automaticRelease = true)
        signAllPublications()
    }
}
