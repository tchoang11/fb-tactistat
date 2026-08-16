"""Model handles resolve through configs/models.yaml, not through call sites."""

from tactistat.llm.registry import (
    STRUCTURED_METHOD,
    ModelSpec,
    chat_model,
    configure_cache,
    parse_handle,
    resolve,
    resolve_role,
    role_handle,
    structured_model,
)

__all__ = [
    "STRUCTURED_METHOD",
    "ModelSpec",
    "chat_model",
    "configure_cache",
    "parse_handle",
    "resolve",
    "resolve_role",
    "role_handle",
    "structured_model",
]
