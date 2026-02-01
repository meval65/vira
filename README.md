# VIRA: Virtual Intelligent Responsive Assistant (Neural Edition)

VIRA adalah sistem operasi kehidupan pribadi (*Personal Life OS*) berbasis kecerdasan buatan yang dirancang dengan **arsitektur biomimetik**. Sistem ini meniru struktur dan fungsi otak manusia untuk menciptakan asisten yang tidak hanya sekadar "menjawab pertanyaan", tetapi juga memiliki ingatan jangka panjang, kecerdasan emosional, kemampuan perencanaan, dan mekanisme koreksi diri (*self-correction*).

Sistem ini berjalan di atas Telegram sebagai antarmuka utama dan dilengkapi dengan Web Dashboard untuk pemantauan aktivitas neural secara *real-time*.

---

## 🧠 Arsitektur Neural (`src/brain`)

Inti dari VIRA bernama **`src/brain`**, yang terdiri dari modul-modul terpisah (lobus) yang saling berkomunikasi melalui **Neural Event Bus** (berbasis Redis atau In-Memory). Berikut adalah bedah detail dari setiap komponen otak VIRA:

### 1. 🌲 Brainstem (Batang Otak)
**Lokasi:** `src/brain/brainstem.py`
*   **Fungsi Utama:** Bootloader, Sistem Saraf Pusat, & Router LLM.
*   **Cara Kerja:**
    *   Menginisialisasi seluruh organ lain saat startup.
    *   Mengelola **Neural Event Bus** untuk komunikasi *asynchronous* antar modul.
    *   **OpenRouter Client dengan Health Check:** Secara otomatis memantau kesehatan model LLM (Latency, Error Rate) dan melakukan *fallback* ke model lain/tier lebih rendah jika model utama *down* atau lambat.
    *   Menjadwalkan *background jobs* (Cerebellum).

### 2. 🏛️ Hippocampus (Memori & Pembelajaran)
**Lokasi:** `src/brain/hippocampus`
*   **Fungsi Utama:** Memori Jangka Panjang, Pengelolaan Pengetahuan (*Knowledge Graph*), & RAG.
*   **Fitur Detail:**
    *   **Vector Search:** Menyimpan memori dalam bentuk vektor (embedding) di MongoDB untuk pencarian berbasis makna (*semantic search*), bukan sekadar kata kunci.
    *   **Knowledge Graph:** Memetakan hubungan antar entitas (misal: `User` - *likes* -> `Coding`, `Coding` - *requires* -> `Coffee`).
    *   **Memory Compression:** Secara otomatis merangkum memori lama agar hemat token namun konteks tetap terjaga.

### 3. 👑 Prefrontal Cortex (Eksekutif & Perencanaan)
**Lokasi:** `src/brain/prefrontal_cortex`
*   **Fungsi Utama:** Pengambilan Keputusan, Perencanaan Tugas (*Planning*), & Penalar Tingkat Tinggi.
*   **Cara Kerja:**
    *   Menerima input mentah dan memutuskan modul mana yang harus merespons.
    *   Memecah instruksi kompleks menjadi langkah-langkah logis (*Step-by-step Plan*).
    *   Menggunakan LLM tercerdas (*Analysis Model*) untuk menangani logika berat.

### 4. 🎭 Amygdala (Emosi & Kepribadian)
**Lokasi:** `src/brain/amygdala`
*   **Fungsi Utama:** Pemrosesan Emosi, Sentimen, & Adaptasi Persona.
*   **Fitur Detail:**
    *   **Mood State:** VIRA memiliki status emosi internal (Happy, Concerned, Neutral, dll) yang berubah berdasarkan interaksi pengguna.
    *   **Empathy Response:** Nada bicara VIRA beradaptasi dengan emosi pengguna (misal: merespons dengan lembut saat pengguna terdeteksi sedih).
    *   **Emotional Decay:** Emosi akan kembali ke netral seiring berjalannya waktu (normalisasi), meniru sifat alami manusia.

### 5. 📡 Thalamus (Relay & Manajemen Konteks)
**Lokasi:** `src/brain/thalamus`
*   **Fungsi Utama:** Filter Informasi, Manajemen Sesi, & *Insight*.
*   **Cara Kerja:**
    *   **Context Window Management:** Memilih potongan riwayat chat yang paling relevan untuk dikirim ke LLM, menjaga efisiensi token dan biaya.
    *   **Insight Generation:** Secara pasif menganalisis percakapan untuk menemukan pola atau fakta baru tentang pengguna tanpa diminta.

### 6. 🛠️ Parietal Lobe (Refleks & Alat)
**Lokasi:** `src/brain/parietal_lobe`
*   **Fungsi Utama:** Eksekusi Alat (*Tool Use*), Kemampuan Sensorik, & Kalkulasi.
*   **Fitur Unggulan ("Reflexes"):**
    *   Modul ini merespons perintah teknis secara deterministik (pasti) dan cepat.
    *   **Sandboxed Python Execution:** Mampu menjalankan kode Python secara **aman di dalam Docker container terisolasi**. Ini memungkinkan VIRA melakukan perhitungan matematika kompleks, analisis data, atau simulasi algoritma tanpa membahayakan sistem inang.
    *   **Real-time Math & Time:** Kalkulasi matematika presisi dan pengecekan waktu lokal.
    *   **Weather Info:** Integrasi API cuaca untuk konteks lingkungan.

### 7. ⚖️ Medulla Oblongata (Fungsi Otonom)
**Lokasi:** `src/brain/medulla_oblongata`
*   **Fungsi Utama:** Penjaga Kestabilan Sistem.
*   **Cara Kerja:**
    *   **Rate Limiting:** Mencegah sistem dibanjiri permintaan (*spam protection*).
    *   Menangani fungsi-fungsi utilitas tingkat rendah yang menjaga agar sistem tetap berjalan ("bernapas").

### 8. ⚙️ Cerebellum (Pemeliharaan & Otomatisasi)
**Lokasi:** `src/brain/cerebellum`
*   **Fungsi Utama:** *Background Tasks*, Koordinasi Rutin, & Pembersihan.
*   **Fitur Detail:**
    *   Menjalankan tugas pemeliharaan rutin (Cron Jobs).
    *   **Nocturnal Consolidation (Siklus Tidur):** Setiap jam 3 pagi, sistem menjalankan "konsolidasi memori" besar-besaran—merapikan database, memperkuat ingatan penting, dan menghapus *noise* dari hari sebelumnya, mirip proses tidur pada manusia.

### 9. 🛡️ Self Correction System (Sistem Imun Kognitif)
**Lokasi:** `src/brain/self_correction.py`
*   **Fungsi Utama:** Pemulihan Kesalahan Otomatis.
*   **Cara Kerja:**
    *   Jika sebuah alat (misal: Kode Python atau Query Database) gagal, sistem ini tidak langsung menyerah.
    *   Ia melakukan **"Critique"**: Menggunakan LLM terpisah untuk menganalisis pesan error.
    *   **Strategy Adjustment:** Memutuskan strategi perbaikan (misal: memperbaiki argumen variabel, mencoba alat alternatif, atau mengubah format data) dan mencoba ulang eksekusi secara otomatis (*auto-retry loop*).

### 10. 👁️ Occipital Lobe (Visualisasi)
**Lokasi:** `src/brain/occipital_lobe`
*   **Fungsi Utama:** Dashboard Antarmuka.
*   **Fitur:**
    *   Menyediakan API dan WebSocket server untuk Web Dashboard (`localhost:5000`).
    *   Memvisualisasikan aktivitas otak, status memori, dan log chat secara *live*.

### 11. 🗣️ Motor Cortex (Eksekusi Gerak)
**Lokasi:** `src/brain/motor_cortex`
*   **Fungsi Utama:** Antarmuka Eksternal (Telegram).
*   **Cara Kerja:**
    *   Menangani input/output pesan Telegram.
    *   Mengatur *chunking* pesan (memecah respons panjang).
    *   Menangani file upload/download dari user.

---

## 🚀 Fitur Unggulan

1.  **Ingatan Abadi (Persistent Memory)**
    VIRA tidak akan lupa. Ia menyimpan preferensi, fakta, dan konteks percakapan dalam jangka panjang menggunakan kombinasi MongoDB dan *Vector Store*.

2.  **Eksekusi Kode Aman (Docker Sandbox)**
    VIRA bukan sekadar chatbot teks. Ia bisa menjadi *Data Analyst* dengan menulis dan *menjalankan* kode Python sungguhan di lingkungan aman untuk menjawab pertanyaan berbasis komputasi.

3.  **Kecerdasan Emosional Dinamis**
    VIRA merespons dengan nuansa emosi yang tepat, membuatnya terasa lebih hidup dibandingkan AI asisten standar yang kaku.

4.  **Resiliensi Tinggi**
    Dengan **Circuit Breaker** pada API LLM dan **Self-Correction Loop** pada eksekusi alat, VIRA dirancang untuk tetap berjalan stabil meskipun terjadi kesalahan jaringan atau halusinasi model.

---

## 🛠 Teknologi

*   **Bahasa Utama:** Python 3.10+
*   **LLM Engine:** OpenRouter (Mendukung Claude 3.5 Sonnet, GPT-4o, Llama 3, DeepSeek, dll).
*   **Database:** MongoDB (Data Dokumen) & Vector Search.
*   **Cache/Bus:** Redis (Opsional) atau In-Memory.
*   **Visualization:** FastAPI & Uvicorn (Dashboard).
*   **Isolation:** Docker (untuk Python Sandbox).
*   **Interface:** Python-Telegram-Bot.

## 📦 Cara Memulai

1.  **Persiapan Environment**
    Salin `.env.example` ke `.env` dan isi kredensial yang dibutuhkan:
    *   `TELEGRAM_BOT_TOKEN`
    *   `OPENROUTER_API_KEY`
    *   `MONGODB_URI`
    *   `(Opsional) METEOSOURCE_API_KEY` untuk fitur cuaca.

2.  **Instalasi Dependensi**
    ```bash
    pip install -r requirements.txt
    ```
    *Pastikan Docker Desktop berjalan jika ingin menggunakan fitur Python Sandbox.*

3.  **Jalankan VIRA**
    ```bash
    python -m src.brain.brainstem
    ```

4.  **Akses Dashboard**
    Buka `http://localhost:5000` di antarmuka web.
