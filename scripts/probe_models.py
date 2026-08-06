#!/usr/bin/env python
"""Verify every model in configs/models.yaml against the live endpoint.

Provider model catalogues drift: IDs get renamed, checkpoints are retired, and
free-tier eligibility changes without notice. Two failures this script exists
to catch, both observed on 2026-08-06 while building the baseline:

  * ``gemini-2.5-flash`` appears in Gemini's own ``/models`` listing but
    answers ``404`` on ``generateContent`` for a free key. Listing a model is
    not the same as being able to call it.
  * Groq's Llama checkpoints reject strict ``json_schema`` decoding with HTTP
    400 while accepting loose ``json_object`` mode, so a router built on the
    stricter assumption fails only at request time.

Discovering either of those in the middle of an evaluation run costs a rerun
and casts doubt on any numbers already collected. Run this first instead:

    python scripts/probe_models.py                 # every provider with a key
    python scripts/probe_models.py --provider groq
    python scripts/probe_models.py --list          # live catalogue, no calls

Exit code is non-zero when a registered model fails or its real JSON
capability contradicts the ``json_mode`` declared in the registry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from tactistat.config import env, load_config, load_model_registry

# A deliberately tiny task shaped like the real router prompt: force the model
# to pick one value from a closed set. Small enough that probing every model
# costs a negligible slice of a free-tier daily quota.
# The literal word "json" must appear in the prompt: OpenAI-compatible servers
# reject `response_format={"type": "json_object"}` outright when it does not.
PROBE_PROMPT = (
    "Classify this football question as exactly one of STAT, TACTICAL, or HYBRID.\n"
    "Question: How many goals did Messi score at the 2022 World Cup?\n"
    'Reply with json of the form {"label": "..."}.'
)

# Two schema dialects, and one object cannot satisfy both:
#   * OpenAI-compatible strict decoding REQUIRES `additionalProperties: false`
#   * Gemini's OpenAPI-subset schema REJECTS that keyword with HTTP 400
# Normalising this split is a core job of the client adapters; the probe has to
# do it too, or it measures its own bug instead of the provider's capability.
_BASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["STAT", "TACTICAL", "HYBRID"]}},
    "required": ["label"],
}
PROBE_SCHEMA_OPENAI: dict[str, Any] = {**_BASE_SCHEMA, "additionalProperties": False}
PROBE_SCHEMA_GEMINI: dict[str, Any] = _BASE_SCHEMA

# Chat-completion probing is meaningless for these; they are speech, safety, or
# embedding endpoints that happen to share the model listing.
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
    """Return ``(actual_mode, detail)`` for an OpenAI-compatible endpoint.

    Walks the capability ladder from strict to loose and reports the strongest
    mode that actually works, rather than the strongest one advertised.
    """
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
    """Return ``(actual_mode, detail)`` for Gemini.

    Gemini's ``response_schema`` is an OpenAPI-3 subset. It rejects
    ``additionalProperties`` outright, which is why the probe schema omits it;
    the production adapter sanitises schemas for the same reason.
    """
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
    try:
        return json.loads(text)["label"]
    except Exception:  # noqa: BLE001
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])["label"]
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
        # A keyless provider is a local server. "Not running" is a setup state,
        # not a broken model, so probe every model behind it only once we know
        # the server answers -- otherwise one stopped daemon reports as N
        # separate model failures and buries the real ones.
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
