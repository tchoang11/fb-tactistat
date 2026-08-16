"""Resolve a ``provider:alias`` handle into a configured LangChain chat model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from tactistat.config import Config, ConfigError, env, load_model_registry

# Measured json_mode (configs/models.yaml) -> with_structured_output method.
STRUCTURED_METHOD = {"schema": "json_schema", "object": "json_mode"}

_CACHE_PATH: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    """One registry entry, flattened with its provider's settings."""

    handle: str
    provider: str
    kind: str
    model_id: str
    json_mode: str
    api_key_env: str | None = None
    base_url: str | None = None
    signup_url: str | None = None
    extra_body: dict[str, Any] | None = None
    fixed_sampling: bool = False

    @property
    def structured_method(self) -> str:
        """The with_structured_output method this model actually supports."""
        method = STRUCTURED_METHOD.get(self.json_mode)
        if method is None:
            raise ConfigError(
                f"{self.handle} declares json_mode={self.json_mode!r} and cannot be used "
                "for a structured-output role. Pick a model whose probed json_mode is "
                "'schema' or 'object' (python scripts/probe_models.py)."
            )
        return method


def parse_handle(handle: str) -> tuple[str, str]:
    """Split ``provider:alias``."""
    if not isinstance(handle, str) or handle.count(":") != 1 or not all(handle.split(":")):
        raise ConfigError(f"Model handle {handle!r} must be of the form 'provider:alias'")
    provider, alias = handle.split(":")
    return provider, alias


def resolve(handle: str, registry: dict[str, Any] | None = None) -> ModelSpec:
    """Look a handle up in configs/models.yaml."""
    provider_name, alias = parse_handle(handle)
    providers = (registry or load_model_registry())["providers"]
    if provider_name not in providers:
        raise ConfigError(
            f"Unknown provider {provider_name!r} in handle {handle!r}. "
            f"Known: {', '.join(sorted(providers))}"
        )
    provider = providers[provider_name]
    models = provider.get("models") or {}
    if alias not in models:
        raise ConfigError(
            f"Unknown model {alias!r} for provider {provider_name!r}. "
            f"Known: {', '.join(sorted(models))}"
        )
    spec = models[alias]
    return ModelSpec(
        handle=handle,
        provider=provider_name,
        kind=provider["kind"],
        model_id=spec["id"],
        json_mode=spec.get("json_mode", "none"),
        api_key_env=provider.get("api_key_env"),
        base_url=provider.get("base_url"),
        signup_url=provider.get("signup_url"),
        extra_body=provider.get("extra_body"),
        fixed_sampling=bool(spec.get("fixed_sampling", False)),
    )


def role_handle(config: Config, role: str) -> str:
    """Read the handle configured for a role such as ``router``."""
    roles = config.section("models")
    if role not in roles:
        raise ConfigError(f"No model configured for role {role!r}. Known: {', '.join(roles)}")
    return roles[role]


def resolve_role(config: Config, role: str) -> ModelSpec:
    return resolve(role_handle(config, role))


def _api_key(spec: ModelSpec) -> str | None:
    if spec.api_key_env is None:
        return None
    key = env(spec.api_key_env)
    if not key:
        raise ConfigError(
            f"{spec.api_key_env} is not set, required by {spec.handle}. "
            f"Add it to .env ({spec.signup_url})."
        )
    return key


def configure_cache(config: Config) -> None:
    """Point the process-global LLM cache at this config's cache directory.

    LangChain's cache is global, so two experiments in one process would
    otherwise share whichever directory was installed first.
    """
    global _CACHE_PATH
    from langchain_core.globals import set_llm_cache

    if not config.get("llm.cache_enabled", False):
        if _CACHE_PATH is not None:
            set_llm_cache(None)
            _CACHE_PATH = None
        return

    cache_dir = config.path("llm.cache_dir")
    database = str(cache_dir / "langchain.sqlite")
    if database == _CACHE_PATH:
        return

    from langchain_community.cache import SQLiteCache

    cache_dir.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=database))
    _CACHE_PATH = database


def _max_tokens(config: Config, role: str) -> int:
    """Per-role budget; thinking models need more than the shared default."""
    by_role = config.get("llm.max_tokens_by_role", {}) or {}
    return int(by_role.get(role, config["llm.max_tokens"]))


def chat_model(config: Config, role: str, **overrides: Any) -> BaseChatModel:
    """Build the chat model configured for a role."""
    spec = resolve_role(config, role)
    configure_cache(config)

    settings: dict[str, Any] = {
        "timeout": config["llm.timeout_seconds"],
        "max_retries": config["llm.max_retries"],
    }
    # Some models expose no sampling controls and warn on every call.
    if not spec.fixed_sampling:
        settings["temperature"] = config["llm.temperature"]
    budget = overrides.pop("max_tokens", None) or _max_tokens(config, role)

    if spec.kind == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=spec.model_id,
            google_api_key=_api_key(spec),
            max_output_tokens=budget,
            **settings,
            **overrides,
        )
    if spec.kind == "openai_compat":
        from langchain_openai import ChatOpenAI

        # Ollama needs no credential but the SDK insists on a non-empty string.
        return ChatOpenAI(
            model=spec.model_id,
            api_key=_api_key(spec) or "not-needed",
            base_url=spec.base_url,
            max_tokens=budget,
            extra_body=spec.extra_body,
            **settings,
            **overrides,
        )
    raise ConfigError(f"Unknown provider kind {spec.kind!r} for {spec.handle}")


def structured_model(
    config: Config,
    role: str,
    schema: type[BaseModel] | dict[str, Any],
    **overrides: Any,
) -> Runnable:
    """Bind a schema using the method this model was measured to support.

    LangChain otherwise picks the method itself, so a model that cannot honour a
    strict schema silently degrades to best-effort JSON.
    """
    spec = resolve_role(config, role)
    model = chat_model(config, role, **overrides)
    return model.with_structured_output(schema, method=spec.structured_method)
