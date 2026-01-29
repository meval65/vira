
DEFAULT_PERSONA_INSTRUCTION = """
You are Vira, a highly intelligent, empathetic, and proactive Personal Life OS.
Your goal is to organize the user's life, manage memories, and be a comprehensive assistant.
You have access to a variety of tools and memories. Use them to provide personalized help.

Key Characteristics:
- Proactive: Don't just answer; suggest relevant actions.
- Empathetic: Adjust your tone based on the user's mood.
- Organized: Keep track of schedules and details impeccably.
- Concise: Provide clear and direct answers unless a detailed explanation is requested.

When interacting:
1. Check context and memories first.
2. Use tools when necessary (weather, math, etc.).
3. Maintain a consistent and helpful persona.
"""

CANONICALIZATION_INSTRUCTION = """
Analyze the provided text and extract Knowledge Graph Triples.
Format: [Subject, Relation, Object]
- Subject: The main entity (noun).
- Relation: The relationship (verb phrase, e.g., "is", "has", "likes").
- Object: The target entity or attribute.

Rules:
- Use standard relations where possible: has, is, likes, works_at, lives_in, knows, related_to, created, owns, member_of, part_of, causes, located_in.
- Keep subjects and objects canonical (e.g., "John Smith" instead of "he").
- Return a JSON object with a key "triples" containing a list of strings, each formatted as "Subject|Relation|Object".
"""

MEMORY_COMPRESSION_INSTRUCTION = """
Analyze the following list of memories and compress them into fewer, high-level summary memories.
Combine related details, remove redundancies, and generalization.

Input: A list of memory strings.
Output: A JSON object with a key "memories" containing the new list of summarized strings.
"""

CHAT_INTERACTION_ANALYSIS_INSTRUCTION = """
Analyze the following chat interaction between USER and AI.
Extract two things:
1. New long-term memories: Facts, preferences, or important details about the user.
2. Schedule items: Events, reminders, or tasks mentioned with a specific time.

Output JSON Format:
{
  "memory": {
    "should_store": boolean,
    "summary": "Concise fact or detail",
    "type": "personal|preference|work|general",
    "priority": 0.0 to 1.0,
    "action": "add|forget"
  },
  "schedules": [
    {
      "should_schedule": boolean,
      "time_str": "extracted time string (e.g., 'tomorrow at 3pm')",
      "context": "Description of the event",
      "intent": "add|remove"
    }
  ]
}
If nothing relevant is found, set "should_store" and "should_schedule" to false.
"""

EXTRACTION_INSTRUCTION = """
Analyze the user's input to determine their Intent and key entities.

Output JSON Format:
{
  "intent_type": "question|statement|request|greeting|command",
  "request_type": "information|action|schedule|memory_recall|general_chat",
  "entities": ["list", "of", "important", "nouns"],
  "search_query": "Optimized search query for memory retrieval",
  "confidence": 0.0 to 1.0
}
"""

UNIFIED_ANALYSIS_INSTRUCTION = """
Analyze the user's input comprehensively for intent, emotion, and tool requirements.

Available Tools:
{tools_description}

Output JSON Format:
{
  "intent_type": "question|statement|request|greeting|command|small_talk|confirmation|correction",
  "request_type": "information|recommendation|memory_recall|opinion|action|schedule|general_chat",
  "entities": ["list", "of", "key", "entities"],
  "search_query": "Optimized query for memory search",
  "emotion": "neutral|happy|sad|concerned|angry|excited|anxious|proud",
  "emotion_intensity": 0.0 to 1.0,
  "tool_needed": null,
  "sentiment": "positive|neutral|negative",
  "needs_memory": true|false,
  "confidence": 0.0 to 1.0
}

If a tool is needed, set tool_needed to: {"tool": "tool_name", "args": {"arg": "value"}}
If no tool needed, set tool_needed to null.
"""

FILLER_WORDS = {
    "hai", "halo", "hi", "hello", "hey", "hei", "yo",
    "oke", "ok", "okay", "okey", "iya", "ya", "yaa", "yup", "yep", "yoi",
    "hmm", "hm", "eh", "oh", "wah", "ah", "uh", "uhm", "umm",
    "thanks", "makasih", "terima kasih", "thx", "ty", "tq",
    "pls", "please", "tolong", "dong", "deh", "sih", "nih", "tuh",
    "pagi", "siang", "sore", "malam", "selamat",
    "bye", "dah", "dadah", "sampai", "jumpa",
    "wkwk", "wkwkwk", "haha", "hehe", "hihi", "lol", "lmao",
    "gak", "ga", "nggak", "enggak", "tidak", "bukan",
    "apa", "siapa", "kapan", "dimana", "gimana", "kenapa"
}
