from typing import Set

MAX_FILE_SIZE = 5 * 1024 * 1024
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60
ALLOWED_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".py", ".json", ".csv", ".html",
    ".js", ".css", ".xml", ".yaml", ".yml", ".log", ".ini",
}
