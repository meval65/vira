# Vira Personal Life OS - Neural Architecture Documentation

## Overview

Vira is a sophisticated Personal Life Operating System designed with a biologically inspired neural architecture. Unlike traditional chatbots, Vira operates as a cohesive system of specialized modules mimicking the human brain's functional regions. This architecture ensures high-performance low-latency execution, persistent memory with semantic retrieval, emotional state continuity, and proactive automated background processing.

Designed exclusively for a single administrator, Vira eliminates multi-user routing overhead, prioritizing deep personalization and system security.

## Neural System Architecture

The system is composed of ten specialized modules, each responsible for distinct cognitive and operational functions:

### 1. Brainstem (`brainstem.py`)
**Role:** Central Nervous System & Entry Point
- **Responsibility:** Bootstraps the application, loads environment configurations, and orchestrates the neural network.
- **Functions:** 
  - Initializes the Telegram application with concurrent update handling.
  - Manages the `APIRotator` for failover logic across multiple API keys and models.
  - Enforces admin-only security access control at the root level.

### 2. Hippocampus (`hippocampus.py`)
**Role:** Long-Term Memory & Knowledge Storage
- **Responsibility:** Manages persistent data storage using asynchronous SQLite with WAL (Write-Ahead Logging) mode.
- **Functions:**
  - **Episodic Memory:** Stores user interactions with semantic fingerprinting to prevent duplication.
  - **Knowledge Graph:** Manages Subject-Predicate-Object (SPO) triples for structured entity relationship tracking.
  - **Schedule Management:** Handles temporal triggers for reminders and tasks.
  - **Consolidation:** Periodically merges short-term buffers into long-term storage.

### 3. Prefrontal Cortex (`prefrontal_cortex.py`)
**Role:** Executive Function & Reasoning
- **Responsibility:** Handles high-level cognitive processing, intent analysis, and response generation.
- **Functions:**
  - **Intent Extraction:** Analyzes user input to determine the required action (e.g., scheduling, memory recall, casual conversation).
  - **Recursive Planning:** Decomposes complex user requests into executable multi-step plans.
  - **Response Generation:** Utilizes Large Language Models (LLM) with automatic failover rotation.
  - **Embedding Generation:** Converts inputs into vector embeddings for semantic search.

### 4. Amygdala (`amygdala.py`)
**Role:** Emotional Processing & Persona Management
- **Responsibility:** Maintains a persistent emotional state that evolves based on interactions.
- **Functions:**
  - **Mood Tracking:** Tracks variables such as Happiness, Satisfaction, and Energy.
  - **Persona Adaptation:** Dynamically adjusts system prompts (System Instructions) based on current mood (e.g., switching to a concerned tone if satisfaction is low).
  - **Decay Logic:** Slowly returns emotional states to neutral over time via background processes.

### 5. Thalamus (`thalamus.py`)
**Role:** Sensory Relay & Context Management
- **Responsibility:** Acts as the traffic controller for incoming information and short-term context window.
- **Functions:**
  - **Session Management:** Maintains the immediate conversation history window.
  - **Context Assembly:** Aggregates relevant memories, schedules, and active plans into a coherent prompt context.
  - **Semantic Retrieval:** Uses Cosine Similarity to retrieve historically relevant messages based on vector embeddings.

### 6. Cerebellum (`cerebellum.py`)
**Role:** Automated Background Processes
- **Responsibility:** Handles asynchronous background tasks without blocking the main cognitive loop.
- **Functions:**
  - **Schedule Checker:** Runs every minute to trigger due reminders.
  - **Memory Optimization:** Periodically deduplicates and indexes memories.
  - **Session Cleanup:** Archives inactive sessions to optimize RAM usage.
  - **Emotional Decay:** Applies temporal decay filters to emotional values.

### 7. Occipital Lobe (`occipital_lobe.py`)
**Role:** System Visibility & Visualization
- **Responsibility:** Provides a Web API and WebSocket interface for the visual dashboard.
- **Functions:**
  - **System Status API:** Exposes endpoints for real-time monitoring of API health, active models, and memory stats.
  - **Data Visualization:** Serves data for the React-based frontend dashboard.
  - **Instruction Override:** Allows runtime modification of the core persona without system restart.

### 8. Motor Cortex (`motor_cortex.py`)
**Role:** Output Execution & Input Handling
- **Responsibility:** Processes raw input signals (text, audio, images) and executes output actions.
- **Audio Processing:** Integrates with Whisper (via Ollama) to transcribe voice messages and audio files.
- **Command Handling:** Maps standardized commands (`/start`, `/status`, `/reset`) to system functions.

### 9. Medulla Oblongata (`medulla_oblongata.py`)
**Role:** Autonomic Utility Functions
- **Responsibility:** Provides low-level survival utilities.
- **Functions:**
  - **Rate Limiting:** Protects the system from request flooding.
  - **File I/O:** Safe handling of file uploads and downloads with size limits.
  - **Sanitization:** Markdown escaping and input validation.

### 10. Dashboard (`src/dashboard/index.html`)
**Role:** User Interface
- **Responsibility:** A single-page React application for monitoring the neural system.
- **Features:** Real-time visualization of brain activity, system health, mood indicators, and memory feed.

---

## Key Technical Features

### Auto-Rotate API System
To ensure 99.9% uptime, the system implements an `APIRotator` class.
- **Logic:** `gemini-2.5-flash` -> `gemini-2.0-flash` -> `gemini-1.5-flash`.
- **Failover:** If a rate limit (HTTP 429) is encountered, the request is instantly retried with the next model in the queue.
- **Key Rotation:** If all models fail, it rotates to the next available Google API Key in the pool.

### Vector Embedding Memory
Instead of simple keyword matching, Vira uses semantic search.
- **Model:** `bge-m3` (via Ollama).
- **Process:** Every message is embedded into a high-dimensional vector.
- **Retrieval:** When a user asks a question, the system calculates Cosine Similarity between the query vector and historical message vectors to retrieve the top 5 most relevant context pieces.

### Proactive Audio Pipeline
The `Motor Cortex` automatically detects voice messages.
- **Pipeline:** Download -> Convert to OGG/MP3 -> Send to local Whisper model -> Inject Transcript into Chat Stream.

---

## Installation & Setup

### Prerequisites
- **Python 3.12+**
- **Ollama** (for local embeddings and audio transcription)
  - Required models: `ollama pull bge-m3` and `ollama pull whisper`
- **Google Gemini API Key** (Multiple keys recommended for rotation)
- **Telegram Bot Token**

### Environment Configuration
Create a `.env` file in the root directory:

```env
# Core Credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
ADMIN_TELEGRAM_ID=your_numeric_telegram_id
GOOGLE_API_KEY=key1,key2,key3

# System Configuration
DB_PATH=storage/memory.db
CHAT_MODEL=models/gemini-2.5-flash
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Installation Steps

1. **Clone the Repository**
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Dependency Note: Ensure `aiohttp`, `python-telegram-bot[job-queue]`, `google-genai`, `aiosqlite`, `fastapi`, `uvicorn`, `pydantic`, `httpx`, `python-dotenv`, `pillow` are installed.*

3. **Initialize Local Models**
   ```bash
   ollama pull bge-m3
   ollama pull whisper
   ```

4. **Launch the System**
   ```bash
   python -m src.brainstem
   ```

## Usage Guide

### Bot Commands
- `/start` - Initialize neural link.
- `/status` - View crude system metrics.
- `/reset` - Clear short-term context (Thalamus reset).
- `/bio [text]` - Update administrative profile in Hippocampus.

### Dashboard Access
When the system starts, the Occipital Lobe initializes a local web server.
- **URL:** `http://localhost:5000`
- **Features:**
  - View real-time mood and satisfaction.
  - Monitor API health and rotation status.
  - override personas in real-time.
  - Browse knowledge graph and memory feed.