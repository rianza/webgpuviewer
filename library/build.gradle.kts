import com.android.build.api.artifact.SingleArtifact
import java.io.File
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

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

val embed: Configuration = configurations.create("embed")
configurations.named("compileOnly") { extendsFrom(embed) }

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.annotation)
    implementation(libs.androidx.core)
    implementation(libs.androidx.compose.foundation)

    embed(libs.androidx.webgpu)
}

androidComponents {
    onVariants(selector().withBuildType("release")) { variant ->
        val mergeTask =
            project.tasks.register<MergeEmbeddedAarsTask>(
                "merge${variant.name.replaceFirstChar { it.uppercase() }}EmbeddedAars"
            ) {
                embedAars.from(embed)
            }
        variant.artifacts
            .use(mergeTask)
            .wiredWithFiles(
                MergeEmbeddedAarsTask::inputAar,
                MergeEmbeddedAarsTask::outputAar
            )
            .toTransform(SingleArtifact.AAR)
    }
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

/**
 * Merges the classes.jar, native libs (jni/), and assets from [embedAars] into [inputAar],
 * producing [outputAar]. Used to embed androidx.webgpu directly into this library's own AAR
 * so downstream consumers don't need to resolve it (or its custom repo) separately.
 */
abstract class MergeEmbeddedAarsTask : DefaultTask() {
    @get:InputFile
    abstract val inputAar: RegularFileProperty

    @get:InputFiles
    abstract val embedAars: ConfigurableFileCollection

    @get:OutputFile
    abstract val outputAar: RegularFileProperty

    @TaskAction
    fun merge() {
        val outFile = outputAar.get().asFile
        outFile.parentFile.mkdirs()
        if (outFile.exists()) outFile.delete()

        val ownAarFile = inputAar.get().asFile
        val embedAarFiles = embedAars.files

        val mergedClassesJarBytes = mergeClassesJars(ownAarFile, embedAarFiles)

        ZipFile(ownAarFile).use { ownZip ->
            ZipOutputStream(outFile.outputStream()).use { zos ->
                val writtenPaths = mutableSetOf<String>()

                fun writeEntry(name: String, bytes: ByteArray) {
                    if (!writtenPaths.add(name)) return
                    zos.putNextEntry(ZipEntry(name))
                    zos.write(bytes)
                    zos.closeEntry()
                }

                for (entry in ownZip.entries()) {
                    if (entry.isDirectory) continue
                    if (entry.name == "classes.jar") {
                        writeEntry("classes.jar", mergedClassesJarBytes)
                    } else {
                        writeEntry(entry.name, ownZip.getInputStream(entry).readBytes())
                    }
                }

                for (embedAarFile in embedAarFiles) {
                    ZipFile(embedAarFile).use { embedZip ->
                        for (entry in embedZip.entries()) {
                            if (entry.isDirectory) continue
                            if (entry.name == "classes.jar") continue
                            if (entry.name == "AndroidManifest.xml") continue
                            writeEntry(entry.name, embedZip.getInputStream(entry).readBytes())
                        }
                    }
                }
            }
        }
    }

    private fun mergeClassesJars(ownAarFile: File, embedAarFiles: Set<File>): ByteArray {
        val tmpJar = File.createTempFile("merged-classes", ".jar")
        try {
            ZipOutputStream(tmpJar.outputStream()).use { zos ->
                val seen = mutableSetOf<String>()

                fun addClassesJarFrom(aarFile: File) {
                    ZipFile(aarFile).use { aarZip ->
                        val classesEntry = aarZip.getEntry("classes.jar") ?: return
                        val tmpIn = File.createTempFile("classes-in", ".jar")
                        try {
                            aarZip.getInputStream(classesEntry).use { input ->
                                tmpIn.outputStream().use { input.copyTo(it) }
                            }
                            ZipFile(tmpIn).use { classesZip ->
                                for (entry in classesZip.entries()) {
                                    if (entry.isDirectory) continue
                                    if (!seen.add(entry.name)) continue
                                    zos.putNextEntry(ZipEntry(entry.name))
                                    classesZip.getInputStream(entry).use { it.copyTo(zos) }
                                    zos.closeEntry()
                                }
                            }
                        } finally {
                            tmpIn.delete()
                        }
                    }
                }

                // Our own classes come first so they win any (unexpected) name conflicts.
                addClassesJarFrom(ownAarFile)
                for (embedAarFile in embedAarFiles) {
                    addClassesJarFrom(embedAarFile)
                }
            }
            return tmpJar.readBytes()
        } finally {
            tmpJar.delete()
        }
    }
}
