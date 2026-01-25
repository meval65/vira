# 🧠 Vira - Personal Life OS

**Vira** adalah asisten AI personal berbasis arsitektur **Neuroscience-Inspired** yang dirancang untuk memahami, mengingat, dan merespons secara kontekstual layaknya manusia. Sistem ini menggunakan MongoDB sebagai "memori jangka panjang" dan menyimulasikan cara kerja otak manusia melalui modul-modul terpisah.

---

## 📐 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TELEGRAM BOT                                │
│                     (User Interface Layer)                           │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          BRAINSTEM                                   │
│              (Core Orchestrator & System Controller)                 │
│  • Mengelola lifecycle semua modul                                   │
│  • Routing pesan dari Telegram ke Prefrontal Cortex                 │
│  • Background jobs (schedule check, proactive check)                │
│  • NeuralEventBus untuk monitoring real-time                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌───────────────┐
│  HIPPOCAMPUS  │◄────►│ PREFRONTAL      │◄────►│   AMYGDALA    │
│  (Memory)     │      │ CORTEX          │      │  (Emotion)    │
│               │      │ (Decision)      │      │               │
│ • Long-term   │      │                 │      │ • Detect      │
│   memory      │      │ • Intent        │      │   emotion     │
│ • Knowledge   │      │   analysis      │      │ • Adjust      │
│   graph       │      │ • Response      │      │   empathy     │
│ • Personas    │      │   generation    │      │ • Track       │
│ • Schedules   │      │ • Embedding     │      │   satisfaction│
└───────────────┘      │   creation      │      └───────────────┘
        ▲              └────────┬────────┘              ▲
        │                       │                       │
        │              ┌────────┴────────┐              │
        │              │    THALAMUS     │              │
        │              │   (Context)     │              │
        │              │                 │              │
        └──────────────│ • Session mgmt  │──────────────┘
                       │ • Hybrid context│
                       │ • Vector search │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  MOTOR CORTEX   │
                       │   (Output)      │
                       │                 │
                       │ • Send response │
                       │ • Format output │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ OCCIPITAL LOBE  │
                       │  (Dashboard)    │
                       │                 │
                       │ • Web UI        │
                       │ • REST API      │
                       │ • WebSocket     │
                       └─────────────────┘
```

---

## 🧬 Modul-Modul Otak

### 1. **Brainstem** (`src/brainstem.py`)
**Fungsi:** Pusat kendali dan orkestrasi seluruh sistem.

| Komponen | Deskripsi |
|----------|-----------|
| `SystemConfig` | Konfigurasi sistem (model, admin ID, interval) |
| `NeuralEventBus` | Event bus untuk monitoring real-time antar modul |
| `APIRotator` | Rotasi API key Gemini untuk menghindari rate limit |
| `BrainStem` | Inisialisasi & shutdown semua modul |

**Aliran Data:**
```
Telegram → Brainstem → PrefrontalCortex → Response → Telegram
```

---

### 2. **Prefrontal Cortex** (`src/prefrontal_cortex.py`)
**Fungsi:** Pengambilan keputusan, analisis intent, dan generasi respons.

| Fitur | Deskripsi |
|-------|-----------|
| Intent Analysis | Mengekstrak intent dan entitas dari pesan |
| Response Generation | Membuat respons menggunakan Gemini LLM |
| Embedding Creation | Generate embedding vektor untuk setiap pesan |
| Plan Execution | Menjalankan rencana multi-langkah |

**Koneksi:**
- **→ Hippocampus:** Query memori terkait
- **→ Amygdala:** Cek emosi dan sesuaikan respons
- **→ Thalamus:** Bangun konteks percakapan
- **→ Motor Cortex:** Kirim output

---

### 3. **Hippocampus** (`src/hippocampus.py`)
**Fungsi:** Penyimpanan dan retrieval memori jangka panjang.

| Collection | Data |
|------------|------|
| `memories` | Fakta, preferensi, event, bio |
| `knowledge_graph` | Triple (subject-predicate-object) |
| `entities` | Register entitas unik |
| `schedules` | Jadwal dan pengingat |
| `personas` | Profil kepribadian AI |
| `admin_profile` | Profil admin |
| `emotional_state` | State emosional AI |

**Algoritma:**
- **Canonicalization:** Konversi teks → struktur (entity, relation, value)
- **Fingerprinting:** Deteksi duplikat memori
- **Memory Merging:** Gabung memori serupa dengan confidence boost
- **Vector Search:** NumPy cosine similarity untuk retrieval semantik

---

### 4. **Amygdala** (`src/amygdala.py`)
**Fungsi:** Deteksi dan respons emosional.

| Metrik | Deskripsi |
|--------|-----------|
| `mood` | Kondisi mental saat ini (happy, sad, neutral, dll) |
| `empathy` | Level empati (0.0 - 1.0) |
| `satisfaction` | Kepuasan interaksi (-1.0 hingga 1.0) |

**Fitur:**
- Deteksi emosi dari teks pengguna
- Adaptasi gaya respons berdasarkan emosi
- Tracking perubahan mood sepanjang waktu

---

### 5. **Thalamus** (`src/thalamus.py`)
**Fungsi:** Manajemen sesi dan pembangunan konteks.

| Fitur | Deskripsi |
|-------|-----------|
| Hybrid Retrieval | 20 pesan terakhir + 5 pesan relevan via vector search |
| Context Building | Gabung memori, jadwal, profil, cuaca → konteks |
| Proactive Insights | Deteksi inactivity, reminder, knowledge gaps |
| Weather Integration | Ambil data cuaca lokal |

**TTL (Time-To-Live):**
- Chat logs otomatis expire setelah **90 hari**

---

### 6. **Occipital Lobe** (`src/occipital_lobe.py`)
**Fungsi:** Visualisasi dan antarmuka manajemen.

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/api/memories` | GET/POST/PUT/DELETE | CRUD memori |
| `/api/chat-logs` | GET/DELETE | Riwayat percakapan |
| `/api/schedules` | GET/POST/DELETE | Jadwal |
| `/api/personas` | GET/POST/PUT/DELETE | Persona AI |
| `/api/profile` | GET/PUT | Profil admin |
| `/api/neural-status` | GET | Status modul real-time |
| `/ws` | WebSocket | Neural activity stream |

---

## 🔄 Aliran Data Pemrosesan Pesan

```
1. USER mengirim pesan via Telegram
   │
   ▼
2. BRAINSTEM menerima & validasi (admin only)
   │
   ▼
3. PREFRONTAL CORTEX memproses:
   │
   ├── 3a. Generate EMBEDDING untuk pesan
   │
   ├── 3b. Ekstrak INTENT (intent type, entities, sentiment)
   │       └── Emit: prefrontal_cortex → amygdala (emotion_check)
   │
   ├── 3c. AMYGDALA deteksi emosi & adjust empathy
   │       └── Return: emotion data
   │
   ├── 3d. Query HIPPOCAMPUS untuk memori relevan
   │       └── Emit: prefrontal_cortex → hippocampus (memory_retrieval)
   │       └── Return: relevant memories
   │
   ├── 3e. THALAMUS bangun konteks hybrid
   │       └── Short-term: 20 pesan terakhir
   │       └── Long-term: 5 pesan similar (vector search)
   │       └── Return: full context
   │
   └── 3f. Generate RESPONSE dengan LLM
           └── Emit: prefrontal_cortex → motor_cortex (output_sent)
   │
   ▼
4. Update SESSION (simpan pesan + respons + embeddings)
   │
   ▼
5. POST-PROCESS (background):
   │
   ├── Analisis respons untuk ekstraksi memori baru
   ├── Update knowledge graph jika ada relasi baru
   └── Simpan emotional state
   │
   ▼
6. RESPONSE dikirim ke Telegram
```

---

## 📊 Database Schema (MongoDB)

### Collections

| Collection | Deskripsi | TTL |
|------------|-----------|-----|
| `memories` | Memori jangka panjang | - |
| `chat_logs` | Riwayat percakapan | 90 hari |
| `knowledge_graph` | Triple relasi | - |
| `entities` | Entitas unik | - |
| `schedules` | Jadwal & reminder | - |
| `personas` | Profil kepribadian | - |
| `admin_profile` | Single document | - |
| `emotional_state` | Single document | - |

### Memory Document
```json
{
  "_id": "uuid",
  "summary": "User suka kopi hitam tanpa gula",
  "type": "preference",
  "priority": 0.8,
  "confidence": 0.9,
  "embedding": [0.12, -0.05, ...],
  "fingerprint": "preference:likes:kopi",
  "entity": "kopi",
  "relation": "likes",
  "created_at": "2024-01-01T12:00:00",
  "last_used": "2024-01-15T09:30:00",
  "use_count": 5,
  "status": "active"
}
```

---

## 🖥️ Dashboard Features

### Brain Activity Monitor
- **Real-time visualization** modul otak yang aktif
- **Animated data flow** antar modul
- **Per-module status** (idle/active + current task)
- **Event log** perpindahan data

### CRUD Operations
- **Memories:** Create, Read, Update, Delete
- **Chat Logs:** Read, Search, Delete
- **Schedules:** Create, Read, Delete
- **Personas:** Create, Activate, Delete
- **Profile:** Read, Update

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- MongoDB Community Edition
- Telegram Bot Token

### Installation

```bash
# Clone repository
cd vira

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env dengan API keys

# Start MongoDB
mongod

# Run migration (jika ada data SQLite lama)
python -m src.migration.migrate_sqlite_to_mongo

# Start bot
python -m src.brainstem
```

### Environment Variables

| Variable | Deskripsi |
|----------|-----------|
| `GOOGLE_API_KEY` | Gemini API key(s), comma-separated |
| `TELEGRAM_BOT_TOKEN` | Token dari @BotFather |
| `ADMIN_TELEGRAM_ID` | Telegram ID admin |
| `MONGO_URI` | MongoDB connection string |
| `MONGO_DB_NAME` | Nama database |
| `CHAT_MODEL` | Model Gemini untuk chat |

---

## 📁 Project Structure

```
vira/
├── src/
│   ├── brainstem.py         # Core orchestrator
│   ├── prefrontal_cortex.py # Decision & response
│   ├── hippocampus.py       # Memory storage
│   ├── amygdala.py          # Emotion processing
│   ├── thalamus.py          # Context management
│   ├── occipital_lobe.py    # Dashboard API
│   ├── db/
│   │   └── mongo_client.py  # MongoDB connection
│   ├── dashboard/
│   │   └── index.html       # Web dashboard
│   └── migration/
│       └── migrate_sqlite_to_mongo.py
├── storage/                  # Local file storage
├── requirements.txt
├── .env
└── README.md
```

---

## 🔧 API Reference

### Memories
```http
GET    /api/memories?limit=50
POST   /api/memories          {"summary": "...", "memory_type": "fact"}
PUT    /api/memories/{id}     {"summary": "..."}
DELETE /api/memories/{id}
```

### Chat Logs
```http
GET    /api/chat-logs?limit=50
GET    /api/chat-logs/search?q=keyword
DELETE /api/chat-logs/{id}
DELETE /api/chat-logs         # Clear all
```

### Schedules
```http
GET    /api/schedules
POST   /api/schedules         {"context": "...", "scheduled_at": "ISO8601"}
DELETE /api/schedules/{id}
```

### Neural Status
```http
GET    /api/neural-status     # Real-time module states
GET    /api/system-status     # System health & config
```

---

## 📜 License

MIT License - Bebas digunakan dan dimodifikasi.

---

## 🧠 Philosophy

Vira dibangun dengan filosofi bahwa AI assistant yang efektif harus:

1. **Mengingat** - Tidak melupakan informasi penting tentang pengguna
2. **Memahami Konteks** - Menghubungkan percakapan masa lalu dengan sekarang
3. **Berempati** - Menyesuaikan gaya komunikasi dengan kondisi emosional
4. **Proaktif** - Mengingatkan tanpa diminta ketika diperlukan
5. **Transparan** - Menyediakan akses penuh ke "isi otaknya"

---

**Built with 💜 for personal productivity**