
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
* `fitted_models.pkl`: File container yang menyimpan state model ensemble terbaru.
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
3. **Adapt:** Algoritma RLS mulai menyesuaikan parameter model berdasarkan data harga *tick* terbaru.
4. **Execute:** Jika filter berita bersih dan sinyal model memenuhi syarat threshold, `trade_engine.py` akan mengirimkan perintah eksekusi.

---

##  Pengembangan Selanjutnya

* [ ] Implementasi Control Panel interaktif pada Dashboard.
* [ ] Optimasi penggunaan memori untuk pemrosesan MTF (Multi-Timeframe) yang lebih ringan.
* [ ] Integrasi notifikasi Telegram untuk peringatan drawdown dan rilis berita penting.

---

**Disclaimer:** *Trading melibatkan risiko yang signifikan. Perangkat lunak ini disediakan hanya untuk tujuan penelitian dan alat bantu analisis. Pengembang tidak bertanggung jawab atas kerugian finansial yang terjadi.*

---

