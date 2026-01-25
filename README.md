# Vira - Personal Life OS

A brain-inspired Personal Life OS built exclusively for a single admin user. This architecture eliminates multi-user overhead for minimal latency execution.

## Neuro-Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BRAINSTEM                                │
│              (Entry Point & System Configuration)               │
│                                                                 │
│  • Telegram Bot Initialization                                  │
│  • Admin-Only Filter                                            │
│  • Background Job Scheduling                                    │
│  • Environment Configuration                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐   ┌───────────────┐
│   THALAMUS    │   │ PREFRONTAL       │   │   AMYGDALA    │
│               │   │ CORTEX           │   │               │
│ Sensory Relay │   │ Executive Control│   │ Emotional     │
│               │   │                  │   │ State         │
│ • Session     │◀─▶│ • Chat Process   │◀─▶│               │
│ • Context     │   │ • Task Planning  │   │ • Mood        │
│ • Proactive   │   │ • Intent Extract │   │ • Satisfaction│
│ • Weather     │   │ • Analysis       │   │ • Persona     │
└───────────────┘   └──────────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   HIPPOCAMPUS    │
                    │                  │
                    │  Long-Term       │
                    │  Memory          │
                    │                  │
                    │ • SQLite DB      │
                    │ • Memories       │
                    │ • Knowledge Graph│
                    │ • Schedules      │
                    │ • Admin Profile  │
                    └──────────────────┘
```

## Brain Modules

### Brainstem (`brainstem.py`)
Central nervous system entry point. Handles:
- Telegram bot initialization with admin-only access
- System configuration (models, API keys, prompts)
- Background job scheduling (proactive checks, cleanup)
- Neural module orchestration

### Hippocampus (`hippocampus.py`)
Long-term memory storage. Features:
- Async SQLite with WAL mode for performance
- Memory storage with fingerprinting & canonicalization
- Knowledge Graph (SPO triples) for entity relationships
- Admin profile management
- Schedule storage

### Prefrontal Cortex (`prefrontal_cortex.py`)
Executive control center. Handles:
- Message processing with intent extraction
- Task planning & goal decomposition
- Memory retrieval based on context
- LLM response generation
- Post-processing analysis for memory/schedule extraction

### Amygdala (`amygdala.py`)
Persistent emotional state. Features:
- Mood tracking persisted across sessions
- "Kakak Perempuan" (Big Sister) persona
- Satisfaction scoring based on task progress
- Emotion detection from admin messages
- Response tone adaptation

### Thalamus (`thalamus.py`)
Sensory relay system. Handles:
- Session history management
- Context building for LLM
- Proactive insight generation
- Weather data integration
- Inactivity detection for re-engagement

## Setup

### Requirements
- Python 3.12+
- Telegram Bot Token
- Google AI API Key (Gemini)

### Environment Variables
```env
TELEGRAM_BOT_TOKEN=your_bot_token
ADMIN_TELEGRAM_ID=your_telegram_user_id
GOOGLE_API_KEY=your_google_ai_key
DB_PATH=storage/memory.db
METEOSOURCE_API_KEY=optional_weather_key
CHAT_MODEL=models/gemini-2.5-flash
```

### Installation
```bash
pip install python-telegram-bot google-genai aiosqlite pydantic python-dateutil httpx pillow python-dotenv
```

### Running
```bash
python -m src.brainstem
```

## Commands
- `/start` - Start conversation
- `/help` - Show help
- `/reset` - Reset chat session
- `/status` - System status
- `/bio [info]` - View/update profile

## Key Features
- **Single Admin**: No multi-user overhead, zero user_id routing
- **Proactive Contact**: Bot initiates messages on inactivity or reminders
- **Persistent Mood**: Emotional state survives restarts
- **Satisfaction Tracking**: Vira feels "proud" or "concerned" based on task progress
- **Knowledge Graph**: Entity relationships via SPO triples
- **Memory Canonicalization**: Fingerprint-based deduplication