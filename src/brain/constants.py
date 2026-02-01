
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


