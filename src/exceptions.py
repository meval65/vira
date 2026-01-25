"""
Vira AI - Custom Exception Classes
Structured error handling for better debugging and error management.
"""


class ViraException(Exception):
    """Base exception for all Vira-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class APIError(ViraException):
    """Errors related to external API calls (Gemini, Ollama, etc.)."""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, details: dict = None):
        super().__init__(message, details)
        self.api_name = api_name
        self.status_code = status_code


class MemoryError(ViraException):
    """Errors related to memory operations (storage, retrieval, embedding)."""
    
    def __init__(self, message: str, user_id: str = None, operation: str = None, details: dict = None):
        super().__init__(message, details)
        self.user_id = user_id
        self.operation = operation


class SchedulerError(ViraException):
    """Errors related to scheduling operations."""
    
    def __init__(self, message: str, user_id: str = None, schedule_id: int = None, details: dict = None):
        super().__init__(message, details)
        self.user_id = user_id
        self.schedule_id = schedule_id


class DatabaseError(ViraException):
    """Errors related to database operations."""
    
    def __init__(self, message: str, query: str = None, details: dict = None):
        super().__init__(message, details)
        self.query = query


class ConfigurationError(ViraException):
    """Errors related to system configuration."""
    pass


class RateLimitError(ViraException):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, user_id: str = None, wait_seconds: int = None):
        super().__init__(message)
        self.user_id = user_id
        self.wait_seconds = wait_seconds


class ValidationError(ViraException):
    """Input validation errors."""
    
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message)
        self.field = field
        self.value = value
