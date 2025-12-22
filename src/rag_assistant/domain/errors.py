"""Custom exceptions."""


class ConfigError(Exception):
    """Raised when configuration loading fails."""


class DatabaseError(Exception):
    """Raised when database operations fail."""
