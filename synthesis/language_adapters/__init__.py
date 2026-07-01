"""Language adapter registry for the synthesis pipeline."""
from .base import LanguageAdapter
from .python_adapter import PythonAdapter
from .go_adapter import GoAdapter
from .typescript_adapter import TypeScriptAdapter

_REGISTRY = {
    "python": PythonAdapter,
    "go": GoAdapter,
    "typescript": TypeScriptAdapter,
    "ts": TypeScriptAdapter,  # alias
}


def get_adapter(language: str, **kwargs) -> LanguageAdapter:
    """Return an adapter instance for the given language name.

    Args:
        language: One of 'python', 'go', 'typescript' (or 'ts').
        **kwargs: Passed to the adapter constructor.

    Raises:
        ValueError: If the language is not supported.
    """
    lang = language.lower().strip()
    if lang not in _REGISTRY:
        supported = ", ".join(sorted(set(_REGISTRY.keys())))
        raise ValueError(f"Unsupported language '{language}'. Supported: {supported}")
    return _REGISTRY[lang](**kwargs)


__all__ = [
    "LanguageAdapter",
    "PythonAdapter",
    "GoAdapter",
    "TypeScriptAdapter",
    "get_adapter",
]
