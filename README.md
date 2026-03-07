
---

# HF MONITOR - Cassandra Project (VPS Edition)

**HF MONITOR** adalah modul eksekusi live dari ekosistem **Cassandra Project**. Repositori ini dirancang khusus untuk menjembatani model regresi statistik kompleks (VARX, DCC-GARCH, Kalman Filter) yang dihasilkan oleh `main.py` agar dapat melakukan eksekusi perdagangan secara real-time di lingkungan VPS.

Fokus utama dari modul ini adalah stabilitas, adaptabilitas parameter secara real-time menggunakan algoritma **Recursive Least Squares (RLS)**, dan manajemen risiko yang ketat.

---

##  Fitur Utama

* **Model Deployment:** Memfasilitasi model ensemble (VARX, DCC-GARCH, Kalman) yang dilatih di lingkungan riset (Colab/Local) untuk berjalan secara live.
* **Adaptive RLS Engine:** Menggunakan algoritma *Recursive Least Squares* (RLS) untuk memperbarui koefisien model secara real-time tanpa perlu melatih ulang seluruh dataset, memastikan model tetap relevan dengan dinamika pasar terbaru.
* **Parameter Monitoring:** Pemantauan detail parameter model secara berkelanjutan untuk memastikan kelayakan statistik sebelum eksekusi order.
* **Risk Management:**
* **News Filter:** Proteksi otomatis terhadap volatilitas tinggi saat rilis berita ekonomi.
* **Drawdown Control:** Pembatasan risiko kerugian maksimal yang terintegrasi.
* **Trade Rules:** Logika eksekusi perdagangan yang kaku dan teruji.


* **Dashboard Monitor:** Antarmuka pemantauan log dan performa engine yang berjalan di VPS.

---

##  Struktur Proyek

* `app.py` / `run.py`: Entry point utama untuk menjalankan engine monitoring dan trading.
* `mt5_adapter.py`: Konektor khusus untuk menjembatani logika Python dengan terminal MetaTrader 5.
* `trade_engine.py`: Inti dari logika perdagangan, manajemen posisi, dan eksekusi.
* `news_manager.py`: Modul pemantau kalender ekonomi dan filter berita.
* `fitted_ensemble.pkl`: File container yang menyimpan state model ensemble terbaru (termasuk struktur per-timeframe dan model DCC-GARCH/Kalman).
* `requirements.txt`: Daftar dependensi Python yang diperlukan.

---

##  Instalasi di VPS

1. **Clone Repositori:**
```bash
git clone https://github.com/bmcs-ux/HF_MONITOR.git
cd HF_MONITOR

```


2. **Setup Virtual Environment:**
```bash
python3 -m venv venv
source venv/bin/activate
# Pastikan rpyc sudah terinstall
pip install rpyc

# Jalankan rpyc_classic server (default port 18812)
rpyc_classic.py --mode=THREADED

```


3. **Instal Dependensi:**
```bash
pip install -r requirements.txt

```


4. **Konfigurasi Environment:**
Pastikan Anda telah mengisi API Key (FRED, dll) dan kredensial MT5 pada file `parameter.py` atau `.env`.

---

##  Alur Kerja (Workflow)

1. **Sync:** File `fitted_ensemble.pkl` dihasilkan oleh `main.py` (Cassandra Core) dan dikirim ke VPS.
2. **Initialize:** `HF MONITOR` memuat model dan memulai koneksi ke provider data/broker.
3. **Adapt:** Algoritma RLS mulai menyesuaikan parameter model berdasarkan data *bar close* terbaru sesuai timeframe model (mis. update H1 dilakukan saat candle H1 close).
4. **Execute:** Jika filter berita bersih dan sinyal model memenuhi syarat threshold, `trade_engine.py` akan mengirimkan perintah eksekusi.

---

## Penjelasan Parameter `parameter.py` Berdasarkan Pemanggilan di `monitor_for_vps.py`

Di bawah ini adalah parameter yang **benar-benar dipakai** oleh `monitor_for_vps.py` beserta fungsi dan efek jika nilainya diubah.

### A) Runtime, path, dan konektivitas

* `ROOT_DIR`, `VPS_PARAM_DIR`, `VPS_DATA_DIR`
  * **Fungsi:** Menentukan root project dan folder referensi untuk import modul + pembacaan file sinkronisasi model/data.
  * **Efek perubahan:** Jika path salah, monitor dapat gagal import modul internal atau gagal memuat data model sehingga siklus eksekusi berhenti lebih awal.
* `FORECAST_OUTPUT_PATH`, `FRED_DATA_PATH`, `FITTED_MODELS_PATH`
  * **Fungsi:** Lokasi file pickle untuk forecast, data eksogen FRED, dan model fit.
  * **Efek perubahan:** Path tidak valid/berisi file lama dapat memicu prediksi dari data usang atau error saat load.
* `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
  * **Fungsi:** Kredensial login MT5 via `MT5Adapter`.
  * **Efek perubahan:** Kredensial salah mengakibatkan tidak ada koneksi broker, sehingga pengambilan posisi/equity/order gagal.
* `COLAB_API_KEY_FOR_MONITOR`, `COLAB_URL_FILE_PATH`
  * **Fungsi:** Otentikasi dan endpoint sinkronisasi dari Colab/ngrok.
  * **Efek perubahan:** API key salah atau URL file tidak sinkron akan memutus alur update data/model dari sumber eksternal.
* `TRADE_ENGINE_API_URL`, `TRADE_ENGINE_API_KEY`
  * **Fungsi:** Endpoint + kunci autentikasi saat monitor mengirim sinyal ke trade engine.
  * **Efek perubahan:** URL salah membuat sinyal tidak pernah sampai; key salah membuat request ditolak.

### B) Data market, horizon, dan struktur model

* `PAIRS`
  * **Fungsi:** Mapping pair internal ke simbol broker (mis. `GBPUSD -> GBPUSDm`).
  * **Efek perubahan:** Simbol yang tidak tersedia di broker membuat data/tick/order pair tersebut gagal diproses.
* `HF_LOOKBACK_DAYS`, `HF_BASE_INTERVAL`
  * **Fungsi:** Menentukan jendela data historis dan granularitas data HF yang diunduh untuk pembentukan fitur log return.
  * **Efek perubahan:** Terlalu kecil => sinyal/statistik kurang stabil; terlalu besar => beban komputasi dan latensi meningkat.
* `VARX_ENDOG_GROUPS`
  * **Fungsi:** Definisi grouping variabel endogen untuk inferensi dan update RLS per grup.
  * **Efek perubahan:** Group yang tidak sinkron dengan model fit akan menyebabkan missing kolom/parameter sehingga forecast memburuk atau gagal.
* `maxlag_test`, `alpha`
  * **Fungsi:** Parameter uji Granger dan fallback jumlah lag saat metadata model tidak tersedia.
  * **Efek perubahan:** `maxlag_test` lebih besar meningkatkan kompleksitas fitur lag; `alpha` lebih ketat menyaring relasi kausal (sinyal bisa makin jarang).

### C) Parameter adaptasi RLS dan gate stabilitas

* `FORGETTING_FACTOR`
  * **Fungsi:** Kecepatan adaptasi RLS pada update parameter `theta`/`P`.
  * **Efek perubahan:** Lebih kecil => lebih reaktif tetapi lebih noisy; mendekati 1 => lebih stabil tetapi lambat beradaptasi.
* `RLS_INITIAL_P_DIAG`
  * **Fungsi:** Skala kovarians awal RLS.
  * **Efek perubahan:** Nilai besar mempercepat adaptasi awal; terlalu besar berisiko overshoot.
* `RLS_DEVIATION_THRESHOLD`, `RLS_DEVIATION_CLOSE_ALL_THRESHOLD`, `_RLS_DEVIATION_THRESHOLD`
  * **Fungsi:** Ambang deviasi parameter untuk skip trade per-pair dan mode proteksi global (close all).
  * **Efek perubahan:** Ambang lebih kecil = sistem lebih defensif (trade lebih sedikit), ambang lebih besar = lebih agresif.
* `RLS_MIN_UPDATES_FOR_CONFIDENCE`, `RLS_CONFIDENCE_ALPHA`, `RLS_CONFIDENCE_ENTRY_THRESHOLD`, `_RLS_CONFIDENCE`
  * **Fungsi:** Menghitung kematangan/kepercayaan model RLS sebelum entry.
  * **Efek perubahan:** Threshold entry lebih tinggi membuat quality gate lebih ketat namun frekuensi entry turun.

### D) Parameter risk sizing dan eksekusi sinyal

* `EQUITY`, `RISK_PER_TRADE_PCT`
  * **Fungsi:** Basis perhitungan lot sizing saat equity akun tidak tersedia dari broker.
  * **Efek perubahan:** Menaikkan `RISK_PER_TRADE_PCT` memperbesar posisi dan drawdown potensial.
* `K_ATR_STOP`, `K_MODEL_STOP`, `SNR_THRESHOLD`, `TP_RR_RATIO`
  * **Fungsi:** Penentu SL/TP dan kualitas sinyal sebelum order dibuat.
  * **Efek perubahan:** Stop multiplier lebih besar memperlebar SL; `SNR_THRESHOLD` tinggi menyaring sinyal lemah; `TP_RR_RATIO` tinggi menuntut reward lebih besar.
* `RLS_SCALING_FACTOR_SL`, `RLS_SCALING_FACTOR_TP`, `RLS_SNR_INCREASE_FACTOR`, `RLS_TP_RR_MIN`, `RLS_SL_MAX_MULTIPLIER`
  * **Fungsi:** Penyesuaian dinamis risk parameter ketika deviasi RLS membesar.
  * **Efek perubahan:** Sistem dapat otomatis memperlebar SL, memperketat TP/RR minimum, dan menaikkan ambang SNR saat model tidak stabil.
* `DCC_RISK_MULTIPLIER`, `DCC_FLIP_EPS_MULTIPLIER`
  * **Fungsi:** Modulator risiko berbasis kondisi korelasi/contagion dari komponen DCC proxy.
  * **Efek perubahan:** Nilai tinggi membuat respons terhadap rejim risiko lintas aset semakin kuat.
* `NEWS`
  * **Fungsi:** Gate untuk menahan keputusan trading ketika filter berita aktif.
  * **Efek perubahan:** `True` cenderung mengurangi eksposur saat event berisiko tinggi.
  * **Observability:** Dashboard kini menampilkan `news_status` dengan countdown menuju High Impact event berikutnya agar waktu aktivasi gate mudah dipantau.
* `MAGIC_NUMBER`
  * **Fungsi:** Identitas order/posisi milik strategi saat query posisi MT5.
  * **Efek perubahan:** Nilai bentrok dengan EA lain dapat mencampur manajemen posisi lintas strategi.

## Kriteria Model yang Bagus agar Stabil Dieksekusi oleh Sistem Ini

Gunakan checklist berikut saat menyiapkan model dari pipeline training sebelum di-deploy ke VPS:

1. **Struktur fitur konsisten dengan runtime**
   * Nama kolom endogen/eksogen dan lag harus identik dengan yang dibaca monitor.
   * Model harus kompatibel dengan grouping `VARX_ENDOG_GROUPS` dan metadata `lags_used`.
2. **Stabil pada update online (RLS-friendly)**
   * Koefisien baseline tidak terlalu ekstrem agar pembaruan RLS tidak meledak pada beberapa siklus pertama.
   * Inovasi residual relatif terkontrol pada data out-of-sample terbaru.
3. **Sinyal memiliki kualitas eksekusi, bukan hanya akurasi arah**
   * Prediksi perlu menghasilkan rasio reward/risk yang realistis terhadap spread + volatilitas pair.
   * Hindari model yang sering flip sinyal pada noise intrabar.
4. **Robust lintas rejim volatilitas**
   * Lolos validasi pada periode normal dan shock (news/high-vol events), terutama karena sistem punya gate deviasi dan filter berita.
5. **Latensi dan footprint sesuai lingkungan VPS**
   * Inference harus ringan dan deterministik agar tidak menunda pengambilan keputusan pada candle close.
6. **Observability baik**
   * Output model menyertakan artefak yang mudah didiagnosis (forecast, parameter, confidence proxy) sehingga keputusan skip/close dapat diaudit.

## Hasil Tinjauan Codebase & Usulan Tugas Prioritas

Berikut 4 tugas yang direkomendasikan (masing-masing satu kategori):

1. **Perbaikan salah ketik**
   * Rapikan istilah internal terkait news gate menjadi konsisten (misalnya label `High Impact` pada log/UI) agar tidak membingungkan saat audit operasional.
2. **Perbaikan bug**
   * Pastikan parser waktu news menggunakan `timezone` yang benar supaya fitur NEWS tidak gagal saat mengevaluasi jendela blokir.
3. **Perbaikan komentar/dokumentasi**
   * Perbarui dokumentasi dashboard agar mencantumkan panel countdown news (`news_status`) sebagai indikator kapan gate aktif/nonaktif.
4. **Peningkatan pengujian**
   * Tambahkan unit test untuk status gate NEWS (restricted + countdown event berikutnya) agar regresi cepat terdeteksi.

---

##  Pengembangan Selanjutnya

* [ ] Implementasi Control Panel interaktif pada Dashboard.
* [ ] Optimasi penggunaan memori untuk pemrosesan MTF (Multi-Timeframe) yang lebih ringan.
* [ ] Integrasi notifikasi Telegram untuk peringatan drawdown dan rilis berita penting.

---

**Disclaimer:** *Trading melibatkan risiko yang signifikan. Perangkat lunak ini disediakan hanya untuk tujuan penelitian dan alat bantu analisis. Pengembang tidak bertanggung jawab atas kerugian finansial yang terjadi.*

---
