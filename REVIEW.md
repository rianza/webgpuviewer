# Code Review + Catatan Instrumentasi Logging (branch `stability`)

Review dilakukan saat menambahkan logging lengkap (`WgvLog`) ke seluruh file.
Tidak ada perubahan perilaku — hanya penambahan/penyatuan log. Temuan di bawah
**tidak diubah**, hanya dicatat sebagai kandidat perbaikan berikutnya.

## Ringkasan perubahan logging

- Fasade baru `log/WgvLog.kt`: tag seragam `WGV.<modul>`, switch global
  (`enabled`/`minLevel`), dan `throttled()` untuk hot path (per-frame /
  per-gesture) agar logcat tidak dibanjiri.
- Semua pemakaian `android.util.Log` yang menyebar dengan tag tidak konsisten
  (`"Renderer"`, `"WebGpuRenderer"`, `"ImagePage"`, `"Trim"`, `"TileRenderer"`)
  dipindahkan ke tag `WGV.*` yang seragam.
- Blok `catch` yang tadinya diam kini melog error (Image cleanup, TileRenderer
  worker, Trim JNI callbacks, WebGPU render loop).
- Kode native (`trim.cpp`, `resize.cpp`) kini memakai `__android_log_print`
  dengan tag `WGV.TrimCpp` / `WGV.ResizeCpp` (CMake sudah me-link `log`).
- Lihat `LOGGING.md` untuk cara memfilter dengan `adb logcat`.

## Temuan review (kandidat perbaikan, tidak diubah di patch ini)

1. **`defaultDeviceLostCallback` / `defaultUncapturedErrorCallback` melempar
   exception dari callback thread.** Keduanya kini melog (`WGV.Render E`)
   *sebelum* melempar, sehingga penyebab terlihat di logcat; tapi throw dari
   executor tetap berisiko crash app. Pertimbangkan menandai device sebagai
   lost dan re-create, alih-alih melempar.

2. **`Mipmap` drawable constructor memakai `tilesize = 4096`**, sementara
   pipeline utama membuat level dengan `tilesize = 2048`. Tidak fatal (grid
   1x1), tetapi inkonsisten dan bisa membingungkan saat membaca log tile.

3. **`RenderPage.drawMaskedRect` / `Draw.circle` / `Draw.rect` membuat uniform
   buffer baru per panggilan** dan tidak di-destroy (bergantung pada siklus
   hidup encoder/queue). Dokumentasi menyebut ini disengaja untuk menghindari
   race `writeBuffer`, tapi volume alokasi per frame layak dipantau lewat log
   `WGV.Draw`.

4. **Sample app (`sample/`) memakai API lama.** `MainActivity` memanggil
   `state.haveNext = true`, `state.render()`, dan `binding.composeView1.renderer`
   — API yang sudah tidak ada di `ImageViewerState` saat ini (`haveNext` kini
   `val` hasil `fetchPage`, render dipicu `invalidate()`/`collect()`). CI tidak
   meng-compile `:sample`, jadi tidak terdeteksi. Logging tetap ditambahkan,
   tapi sample perlu di-refresh terpisah. Juga tergantung artifact eksternal
   `ca.mpreg.imagedecoder` yang tidak ada di repositori ini.

5. **`ImageViewerState.pageOffset` setter** membaca/menulis `haveNext`/`havePrev`
   yang tiap akses memanggil `fetchPage` (bisa memicu kerja di sisi app) di
   dalam loop `while`. Logging page turn (`WGV.Viewer I: Page turn`) akan
   menunjukkan bila ini terjadi lebih sering dari dugaan.

6. **`WebGpuRenderer.init/cleanup` mendeteksi deadlock lewat nama thread**
   (`"WebGPU-Render-Thread"`). Rapuh terhadap rename; dicatat agar terlihat di
   log thread saat init (`thread=...`).

7. **`Trim.detectBackgroundCpu` vs `detectBackgroundInContext`**: dua jalur
   dengan fallback berbeda (putih). Keduanya kini melog hasilnya (`#RRGGBB`)
   sehingga divergensinya mudah dilihat.

8. **`Image.trim` dipilih berdasarkan area terkecil** di antara warna trim;
   jika warna yang salah menang, log `Trim: winning rect` + `background=#...`
   di `WGV.Image` memudahkan diagnosis.

## Peta titik log penting (debug cepat)

| Gejala | Tag yang diperiksa |
|---|---|
| Layar hitam / surface gagal | `WGV.Render`, `WGV.Gesture` (Surface created/destroyed) |
| Gambar tidak muncul / lambat decode | `WGV.Image`, `WGV.Mipmap`, `WGV.ResizeCpp` |
| Tile tajam lambat muncul | `WGV.Tiles` (worker, batch, evict) |
| Transisi aneh / cache salah | `WGV.Transition` + tag transition konkret |
| Trim salah potong | `WGV.Trim`, `WGV.TrimCpp` |
| Gesture tidak responsif | `WGV.Gesture` |
| Crash saat cleanup | `WGV.Page`, `WGV.Image`, `WGV.Tiles` |
