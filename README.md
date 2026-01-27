# VIRA Personal Life OS (Neural Edition)

VIRA (Virtual Intelligent Responsive Assistant) adalah sistem operasi kehidupan pribadi berbasis AI yang dirancang dengan arsitektur neural modular untuk meniru fungsi otak manusia. Sistem ini mengintegrasikan manajemen memori jangka panjang, kecerdasan emosional, perencanaan tugas otentik, dan antarmuka dashboard real-time.

## 🧠 Arsitektur Neural

Sistem dibangun di atas modul-modul independen yang berkomunikasi melalui **Neural Event Bus** (`brainstem.py`), memungkinkan aliran data yang reaktif dan terorganisir.

### 🧩 Modul Utama (Biological Mapping)

| Modul Biologis | File Source | Fungsi Utama |
| :--- | :--- | :--- |
| **Brainstem** | `src/brainstem.py` | Pusat Inisialisasi, Konfigurasi, & Event Bus (Saraf Pusat). |
| **Hippocampus** | `src/hippocampus.py` | Sistem Memori Jangka Panjang (MongoDB), Knowledge Graph, & RAG. |
| **Prefrontal Cortex** | `src/prefrontal_cortex.py` | Pengambilan Keputusan, Perencanaan (LLM), & Pemrosesan Input. |
| **Amygdala** | `src/amygdala.py` | Pemrosesan Emosi, Sentimen, & Adaptasi Kepribadian. |
| **Thalamus** | `src/thalamus.py` | Manajemen Konteks, Routing Pesan, & Filter Riwayat Chat. |
| **Cerebellum** | `src/cerebellum.py` | Tugas Latar Belakang (Background Jobs), Maintenance, & Otomatisasi. |
| **Motor Cortex** | `src/motor_cortex.py` | Antarmuka Eksekusi (Telegram Bot Handlers & Output). |
| **Occipital Lobe** | `src/occipital_lobe.py` | Visualisasi Sistem (Dashboard API & WebSocket). |
| **Medulla Oblongata** | `src/medulla_oblongata.py` | Utilitas Dasar, Rate Limiting, & File Handling. |

---

## 📂 Struktur File & Tanggung Jawab

### `/src`

*   **`brainstem.py`**
    *   Menginisialisasi seluruh sistem (Bootloader).
    *   Mengelola koneksi ke OpenRouter API.
    *   Menyediakan `NeuralEventBus` untuk komunikasi antar modul secara asynchronous.

*   **`hippocampus.py`**
    *   Menyimpan memori di MongoDB (`memories`).
    *   Mengelola Knowledge Graph (`knowledge_graph`) untuk relasi entitas.
    *   Melakukan pencarian vektor (Vector Search) untuk RAG (Retrieval-Augmented Generation).
    *   Menangani kompresi memori otomatis untuk menghemat konteks.

*   **`prefrontal_cortex.py`**
    *   Otak utama yang memproses input pengguna.
    *   Mengintegrasikan memori, konteks, dan emosi untuk menghasilkan respons.
    *   Menangani perencanaan tugas kompleks (Task Planning).
    *   Ekstraksi intent (Maksud) dari percakapan.

*   **`amygdala.py`**
    *   Menganalisis sentimen pengguna.
    *   Menyimpan state emosi internal (Satisfaction, Mood).
    *   Memodifikasi instruksi prompt berdasarkan mood saat ini (e.g., Happy vs. Sad).

*   **`thalamus.py`**
    *   Menyusun "Prompt Context Window" yang efisien.
    *   Memfilter riwayat chat yang relevan menggunakan embedding.
    *   Mencegah overload token pada LLM.

*   **`cerebellum.py`**
    *   Menjalankan *Cron Jobs* (Tugas berkala).
    *   Pembersihan sesi lama, pengecekan jadwal pengingat, dan optimasi database.
    *   Berjalan di thread terpisah agar tidak memblokir chat utama.

*   **`motor_cortex.py`**
    *   Handler untuk perintah Telegram (`/start`, `/status`, dll).
    *   Mengelola input teks, foto, dan dokumen dari pengguna.
    *   Mengirim respons balik ke pengguna (termasuk chunking pesan panjang).

*   **`occipital_lobe.py`**
    *   Menjalankan server FastAPI untuk Dashboard Lokal (`localhost:5000`).
    *   Websocket server untuk visualisasi aktivitas neural realtime.
    *   API endpoint untuk manajemen memori dan persona via Web UI.

*   **`medulla_oblongata.py`**
    *   Fungsi bantu (Helper functions).
    *   Manajemen unduhan file sementara.
    *   Rate limiting untuk mencegah spam.

---

## ✨ Fitur Utama

1.  **Memori Jangka Panjang (Persistent Memory)**
    *   Vira mengingat fakta, preferensi, dan peristiwa masa lalu menggunakan MongoDB.
    *   Sistem pencarian *semantic* (bukan hanya keyword) untuk konteks yang lebih baik.

2.  **Kecerdasan Emosional**
    *   Vira memiliki "perasaan" yang berubah berdasarkan interaksi.
    *   Respon tone berubah (misal: lebih empatik saat pengguna sedih).

3.  **Manajemen Jadwal & Pengingat**
    *   Deteksi otomatis jadwal dari percakapan ("Ingatkan rapat besok jam 9").
    *   Notifikasi proaktif via Telegram.

4.  **Dashboard Visual Real-time**
    *   Pantau apa yang "dipikirkan" Vira melalui Web Dashboard.
    *   Lihat grafik memori, log chat, dan status emosi secara langsung.

5.  **Multi-Model LLM Support**
    *   Terintegrasi penuh dengan **OpenRouter** (Claude, GPT-4, Llama, Gemini, dll).
    *   Sistem *Fallback* otomatis jika satu model gagal/down.

6.  **Task & Implementation Planning**
    *   Mampu memecah permintaan kompleks menjadi rencana langkah demi langkah.

---

## 🚀 Cara Menjalankan

1.  **Persiapan Environment**
    Pastikan `.env` sudah dikonfigurasi dengan:
    *   `TELEGRAM_BOT_TOKEN`
    *   `ADMIN_TELEGRAM_ID`
    *   `OPENROUTER_API_KEY`
    *   `MONGODB_URI`

2.  **Jalankan Sistem**
    Gunakan perintah berikut di terminal:
    ```bash
    python -m src.brainstem
    ```

3.  **Akses Dashboard**
    Buka browser dan navigasi ke:
    `http://localhost:5000`

---

## 🛠 Tech Stack
*   **Language**: Python 3.10+
*   **Interface**: Python-Telegram-Bot
*   **Database**: MongoDB (Motor Async Driver)
*   **LLM Provider**: OpenRouter API
*   **Backend API**: FastAPI (untuk Dashboard)
*   **Embedding**: Ollama (Local) / Remote Fallback
