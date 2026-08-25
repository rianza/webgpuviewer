# Logging Guide (WebGPUViewer)

Semua log library melewati fasad **`ca.mpreg.webgpuviewer.log.WgvLog`** dengan tag
seragam berawalan `WGV.` — jadi seluruh library bisa difilter dengan satu perintah
`adb logcat`.

## Tag per modul

| Tag | Modul |
|---|---|
| `WGV.Render` | `WebGpuRenderer` — bootstrap WebGPU (instance/adapter/device), surface, frame |
| `WGV.Image` | `Image` — decode pipeline, trim, background, mipmap, upload GPU |
| `WGV.Mipmap` | `Mipmap` — level mipmap, tile upload, uniform |
| `WGV.Tiles` | `TileRenderer` — worker tile, LRU/retain eviction, stencil, blit |
| `WGV.RenderPage` | `RenderPage` — pipeline variants, draw gambar |
| `WGV.Viewer` | `ImageViewerState` — lifecycle, page turn, snapshot |
| `WGV.Continuous` | `ImageViewerContinuousState` — scroll, snapshot kontinu |
| `WGV.Gesture` | `ImageViewer`/`ImageViewerContinuous` + `extensions` — semua gesture |
| `WGV.Page` | `ImagePage` — page Images/Render, animasi frame, cleanup |
| `WGV.Transition` | `Transition` — cache texture transition |
| `WGV.Basic`, `WGV.Cube`, … | masing-masing transition konkret (12 buah) |
| `WGV.Draw` | `Draw`/`clear`/`circle`/`line`/`rect` |
| `WGV.Text` | `Font`/`Draw.text` — msdf atlas |
| `WGV.Trim` | `Trim` + `TrimNative` (Kotlin side) |
| `WGV.Resize` | `ImageUtil` (Kotlin side) |
| `WGV.TrimCpp` | `trim.cpp` (native/JNI) |
| `WGV.ResizeCpp` | `resize.cpp` (native/JNI) |
| `WGV.View` | `ImageView`/`ImageViewContinuous` (entry point View) |
| `WGV.Sample` | aplikasi contoh (`sample/`) |

## Melihat log dengan `adb logcat`

Semua tag sekaligus (pakai grep karena `-s` butuh tag eksak):

```bash
adb logcat | grep -E "WGV\."
```

Modul tertentu (`-s` menyaring exact-tag, boleh lebih dari satu):

```bash
# renderer + viewer + page
adb logcat -s WGV.Render:V WGV.Viewer:V WGV.Page:V

# khusus debugging tile
adb logcat -s WGV.Tiles:V WGV.Mipmap:V WGV.Image:V

# gesture saja
adb logcat -s WGV.Gesture:V

# termasuk log native (C++)
adb logcat -s WGV.TrimCpp:V WGV.ResizeCpp:V
```

Simpan ke file untuk dianalisis / dilampirkan ke issue:

```bash
adb logcat -v time | grep -E "WGV\.|FATAL|AndroidRuntime" > wgv-log.txt
```

Reset buffer lalu rekam sesi bersih:

```bash
adb logcat -c && adb logcat -v time | grep "WGV\."
```

## Level log

`WgvLog.minLevel` default `Log.VERBOSE` (semuanya tampil). Untuk meredam:

```kotlin
WgvLog.minLevel = Log.INFO   // sembunyikan verbose/debug
WgvLog.enabled  = false      // matikan total (paling murah)
```

Log di hot path (per-frame / per-gesture) memakai `WgvLog.throttled(...)`:
maksimal 1 baris per 250–500 ms per (tag, key), dan string-nya tidak
diformat sama sekali saat sedang di-throttle.

## Contoh alur log normal (halaman dibuka, flip, kembali)

```text
WGV.Render  I: WebGPU bootstrap: initLibrary() + createInstance()...
WGV.Render  I: GPUInstance created
WGV.Render  I: GPUAdapter acquired (featureLevel=Compatibility)
WGV.Render  I: Requesting GPUDevice (requiredFeatures=[TimestampQuery])
WGV.Render  I: GPUDevice ready - WebGPU initialised
WGV.Gesture I: Surface created: 1080x2400
WGV.Viewer  I: ImageViewerState.init: 1080x2400, vertical=false, reversed=false
WGV.Image   I: Creating image 1404x1872 (mipmaps=true, trimColors=0, ...)
WGV.TrimCpp I: detectBackground: 1404x1872
WGV.TrimCpp I: detectBackground: solid edges=4 (white=4, nonWhite=0)
WGV.Mipmap  D: create: level 1404x1872 scale=1.0 grid=1x1 (tile=2048)
WGV.Mipmap  D: upload: tile [0,0] 1404x1872 at (0,0)
WGV.Image   I: Image 1404x1872 ready on GPU (1 level(s))
WGV.Viewer  D: Page turn: delta=1 (...)
WGV.Basic    D: Basic: render frac=0.51
...
```
