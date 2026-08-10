#!/usr/bin/env python
"""Verify registered models and their declared JSON capabilities."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from tactistat.config import env, load_config, load_model_registry

# Small router-shaped probe; some endpoints require "json" in the prompt.
PROBE_PROMPT = (
    "Classify this football question as exactly one of STAT, TACTICAL, or HYBRID.\n"
    "Question: How many goals did Messi score at the 2022 World Cup?\n"
    'Reply with json of the form {"label": "..."}.'
)

# OpenAI requires, while Gemini rejects, ``additionalProperties``.
_BASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["STAT", "TACTICAL", "HYBRID"]}},
    "required": ["label"],
}
PROBE_SCHEMA_OPENAI: dict[str, Any] = {**_BASE_SCHEMA, "additionalProperties": False}
PROBE_SCHEMA_GEMINI: dict[str, Any] = _BASE_SCHEMA

# Ignore non-chat endpoints returned by model catalogues.
NON_CHAT_HINTS = ("whisper", "tts", "guard", "embedding", "orpheus", "image", "robotics")


@dataclass
class ProbeResult:
    handle: str
    ok: bool
    latency_ms: int
    declared_mode: str
    actual_mode: str
    detail: str

    @property
    def mode_matches(self) -> bool:
        return not self.ok or self.declared_mode == self.actual_mode


def _probe_openai_compat(base_url: str, api_key: str | None, model_id: str) -> tuple[str, str]:
    """Return the strongest working JSON mode for an OpenAI-compatible model."""
    from openai import OpenAI

    # Ollama needs no credential but the SDK insists on a non-empty string.
    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    attempts = [
        (
            "schema",
            {
                "type": "json_schema",
                "json_schema": {"name": "probe", "schema": PROBE_SCHEMA_OPENAI, "strict": True},
            },
        ),
        ("object", {"type": "json_object"}),
        ("none", None),
    ]
    last_error = ""
    for mode, response_format in attempts:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "temperature": 0,
            "max_tokens": 300,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            reply = client.chat.completions.create(**kwargs)
            text = reply.choices[0].message.content or ""
            label = _extract_label(text)
            if label is None:
                last_error = f"unparseable reply: {text[:60]!r}"
                continue
            return mode, label
        except Exception as exc:  # noqa: BLE001 - report whatever the provider said
            last_error = f"{type(exc).__name__}: {str(exc)[:80]}"
    raise RuntimeError(last_error or "all modes failed")


def _probe_gemini(api_key: str, model_id: str) -> tuple[str, str]:
    """Probe Gemini with its supported OpenAPI schema subset."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    try:
        reply = client.models.generate_content(
            model=model_id,
            contents=PROBE_PROMPT,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=PROBE_SCHEMA_GEMINI,
            ),
        )
        label = _extract_label(reply.text or "")
        if label is None:
            raise RuntimeError(f"unparseable reply: {(reply.text or '')[:60]!r}")
        return "schema", label
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:80]}") from exc


def _extract_label(text: str) -> str | None:
    """Pull ``label`` out of a reply that may or may not be clean JSON."""
    allowed = {"STAT", "TACTICAL", "HYBRID"}

    def valid_label(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        label = payload.get("label")
        return label if label in allowed else None

    try:
        return valid_label(json.loads(text))
    except Exception:  # noqa: BLE001
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return valid_label(json.loads(text[start : end + 1]))
            except Exception:  # noqa: BLE001
                return None
        return None


def _local_server_up(base_url: str, timeout: float = 2.0) -> bool:
    """Cheap liveness check for a keyless local endpoint such as Ollama."""
    import requests

    if not base_url:
        return False
    try:
        requests.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        return True
    except requests.RequestException:
        return False


def list_live_models(provider_name: str, provider: dict[str, Any]) -> list[str]:
    """Ask the provider what it currently serves."""
    import requests

    key_env = provider.get("api_key_env")
    api_key = env(key_env) if key_env else None

    if provider["kind"] == "gemini":
        response = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key or ""},
            timeout=30,
        )
        response.raise_for_status()
        return sorted(
            m["name"].removeprefix("models/")
            for m in response.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        )

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = requests.get(f"{provider['base_url']}/models", headers=headers, timeout=30)
    response.raise_for_status()
    return sorted(m["id"] for m in response.json()["data"])


def probe_provider(provider_name: str, provider: dict[str, Any]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    key_env = provider.get("api_key_env")
    api_key = env(key_env) if key_env else None

    for alias, spec in provider.get("models", {}).items():
        handle = f"{provider_name}:{alias}"
        model_id = spec["id"]
        declared = spec.get("json_mode", "none")
        started = time.time()
        try:
            if provider["kind"] == "gemini":
                actual, detail = _probe_gemini(api_key or "", model_id)
            else:
                actual, detail = _probe_openai_compat(provider["base_url"], api_key, model_id)
            results.append(
                ProbeResult(
                    handle, True, int((time.time() - started) * 1000), declared, actual, detail
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                ProbeResult(
                    handle, False, int((time.time() - started) * 1000), declared, "-", str(exc)[:70]
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", help="probe only this provider")
    parser.add_argument(
        "--list", action="store_true", help="print each provider's live catalogue and exit"
    )
    args = parser.parse_args()

    load_config()  # side effect: loads .env
    registry = load_model_registry()
    providers = registry["providers"]

    if args.provider:
        if args.provider not in providers:
            print(
                f"Unknown provider {args.provider!r}. Known: {', '.join(providers)}",
                file=sys.stderr,
            )
            return 2
        providers = {args.provider: providers[args.provider]}

    reachable: dict[str, dict[str, Any]] = {}
    for name, provider in providers.items():
        key_env = provider.get("api_key_env")
        if key_env and not env(key_env):
            print(f"skip {name:12} {key_env} not set  ({provider['signup_url']})")
            continue
        # Skip a keyless local provider when its server is offline.
        if not key_env and not _local_server_up(provider.get("base_url", "")):
            url = provider.get("base_url")
            print(f"skip {name:12} no server at {url}  ({provider['signup_url']})")
            continue
        reachable[name] = provider

    if not reachable:
        print(
            "\nNo provider has credentials. Copy .env.example to .env and fill in at least one key."
        )
        return 1

    if args.list:
        for name, provider in reachable.items():
            print(f"\n=== {name} live catalogue ===")
            try:
                for model_id in list_live_models(name, provider):
                    if not any(hint in model_id.lower() for hint in NON_CHAT_HINTS):
                        print(f"  {model_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR {type(exc).__name__}: {str(exc)[:100]}")
        return 0

    print(f"\n{'handle':28} {'ok':4} {'ms':>7}  {'declared':9} {'actual':9} detail")
    print("-" * 96)

    all_results: list[ProbeResult] = []
    for name, provider in reachable.items():
        for result in probe_provider(name, provider):
            all_results.append(result)
            flag = "!" if not result.mode_matches else " "
            print(
                f"{result.handle:28} {'OK' if result.ok else 'FAIL':4} {result.latency_ms:>7}  "
                f"{result.declared_mode:9} {result.actual_mode:9}{flag} {result.detail}"
            )

    failures = [r for r in all_results if not r.ok]
    mismatches = [r for r in all_results if not r.mode_matches]

    print()
    print(f"{len(all_results) - len(failures)}/{len(all_results)} models reachable")
    if mismatches:
        print(f"\n{len(mismatches)} json_mode mismatch(es) -- update configs/models.yaml:")
        for r in mismatches:
            print(f"  {r.handle}: declared {r.declared_mode!r}, actually {r.actual_mode!r}")
    if failures:
        print(f"\n{len(failures)} unreachable model(s):")
        for r in failures:
            print(f"  {r.handle}: {r.detail}")

    return 1 if (failures or mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
