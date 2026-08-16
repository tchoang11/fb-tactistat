# TactiStat

An agentic RAG system that answers football questions by routing them to the
right source: computed statistics for numbers, retrieved text for explanations,
and both when a question needs both.

Built on the 2022 FIFA World Cup — 64 matches of StatsBomb event data and 318
Wikipedia articles scoped to the teams, players, and concepts that appear in it.

> **Status: in progress.** The data layer, the stats tool, retrieval, and the
> full question-answering pipeline are complete and tested. The LangGraph graph
> and the evaluation study are being built next — see [Roadmap](#roadmap).

---

## Why this design

A language model asked *"how many goals did Messi score at the 2022 World Cup?"*
will produce a number. Often the right one, sometimes not, and there is no way
to tell which from the answer alone. Asked *"why did Morocco's press work
against Spain?"*, a SQL-style stats bot has nothing to say at all.

The two question types need different machinery:

| Approach | Failure mode |
| --- | --- |
| Retrieval-only RAG | Numbers get paraphrased out of prose and drift |
| Text-to-SQL / stats bot | Correct numbers, no explanation of *why* |
| **TactiStat** | Route by question type, compute numbers, retrieve prose, cite both |

```
                        user question
                              │
                       ┌──────▼──────┐
                       │   Router    │  STAT / TACTICAL / HYBRID
                       └──┬───────┬──┘
                 ┌────────┘       └────────┐
          ┌──────▼──────┐          ┌───────▼───────┐
          │ Stats tool  │          │   RAG tool    │
          │ pandas over │          │ embeddings +  │
          │ 234k events │          │ BM25 over 4.3k│
          │             │          │ Wikipedia     │
          │             │          │ sections      │
          └──────┬──────┘          └───────┬───────┘
                 └────────┐       ┌────────┘
                       ┌──▼───────▼──┐
                       │  Synthesis  │  answer + per-claim citations
                       └─────────────┘
```

Numbers come from a computation over the event table, never from the model's
memory. Explanations come from retrieved text with a page and section
attached. The evaluation grades each stage separately, so a regression can be
attributed rather than guessed at.

---

## Quickstart

Requires Python 3.10–3.12. The steps below use [uv](https://docs.astral.sh/uv/);
plain `venv` + `pip` works equally well.

```bash
git clone https://github.com/tchoang11/fb-tactistat.git
cd fb-tactistat

# Environment
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv
uv venv --python 3.12
uv pip install -e ".[dev]"

# Query the committed processed tables; no API key or raw data needed
tactistat stats goals Messi
tactistat stats goals Messi Mbappe --per90

# Retrieval needs an index built from the committed corpus (~1 min, once)
tactistat build-index
tactistat search "Morocco Spain round of 16 2022" --mode hybrid --rerank

# Run the regression suite
pytest tests/ -q

# Optional: requires the built index and downloaded Hugging Face models
TACTISTAT_RUN_RAG_INTEGRATION=1 pytest tests/test_rag_tool.py -m integration -q
```

The Wikipedia corpus and the aggregated stats tables are **committed to the
repository**. To reproduce them from the raw event data:

```bash
python scripts/01_build_dataset.py          # ~30 s; writes gitignored data/raw/
python scripts/02_build_corpus.py --force   # ~90 s
python scripts/03_build_stats.py            # ~10 s
python scripts/03_build_stats.py --check    # minutes reconciliation only
```

### API keys

None are needed for the data, indexing, or retrieval-only evaluation steps.
Router, synthesis, and the LLM judge need at least one provider:

```bash
cp .env.example .env      # then fill in one or more keys
python scripts/probe_models.py
```

`probe_models.py` calls every configured model once and reports which are
reachable and what output format they really support. Run it before an
evaluation, not during one.

Both defaults are free and need no credit card:

| Provider | Used for | Free tier | Sign-up |
| --- | --- | --- | --- |
| Groq | router, synthesis | ~30 RPM, ~1k req/day per model | <https://console.groq.com/keys> |
| Google Gemini | translator, LLM judge | flash tier | <https://aistudio.google.com/apikey> |

Alibaba Cloud DashScope, Cerebras, OpenRouter, GitHub Models, OpenAI, and a
local Ollama server are also registered in
[`configs/models.yaml`](configs/models.yaml).

---

## Choosing models

Every model is addressed as `provider:alias` and resolved through
[`configs/models.yaml`](configs/models.yaml) into a LangChain chat model, so
adding a provider is a YAML edit. The registry carries a **measured**
`json_mode` per model rather than an assumed one, and the router asks for that
decoding mode explicitly: on Groq, only the `gpt-oss` family accepts strict
`json_schema`, while the Llama checkpoints reject it with HTTP 400 and honour
`json_object`. Letting the framework pick silently would turn "an invalid
label is impossible" into "an invalid label is unlikely" — a difference that
surfaces in the evaluation numbers rather than as an exception.

That measured-not-assumed habit is also what makes a provider retiring a model
a config edit. When Groq decommissioned `llama-3.3-70b-versatile`, the two
suggested replacements were re-measured on the real synthesis prompts rather
than adopted on advice: one of them leaks its reasoning trace into every reply
and was rejected. Swapping the survivor in was a one-line change to
[`configs/default.yaml`](configs/default.yaml).

Each role can point at a different model and can be overridden per run. The
evaluation harness is a roadmap item; the CLI already answers questions end to
end, and exposes the stats engine deterministically:

```bash
tactistat stats goals Messi
tactistat stats goals Messi Mbappe --per90
tactistat stats goals --top-n 5 --stage "Group Stage"

# End to end, in either language, with every stage's decision shown
tactistat ask "Argentina ghi bao nhiêu bàn?" --trace
tactistat ask "Why was Morocco hard to break down?" --strategy hyde
```

A ranking and a total are different questions, and the router picks between
them: Argentina's top five scorers sum to 14 of the squad's 15 goals, so
answering "how many goals did Argentina score?" from a truncated leaderboard is
wrong by one goal. The router fills an `operation` slot — `player`, `compare`,
`ranking` or `total` — and every slot it returns is validated against the
configured metrics before a tool sees it; each repair is recorded so the
evaluation can report which one fired.

The judge deliberately defaults to a different provider *and* model family from
synthesis. LLM judges show a measurable preference for text produced by their
own family, and letting one model both write and grade an answer would put that
bias inside every faithfulness number in the report.

---

## Query translation

Questions arrive in Vietnamese or English, often as a fragment — *"Messi bàn
thắng"*. The corpus is English and BM25 matches literal terms, so an
untranslated question retrieves nothing at all. Translation is therefore a
correctness step, not a tuning knob, and it is kept separate from the rewrite
strategy layered on top of it. Both collapse into a single LLM call.

| `query_translation.strategy` | What is retrieved |
| --- | --- |
| `off` | a literal translation, wording and ambiguity preserved |
| `rewrite` | one self-contained English question |
| `multi_query` | the question plus `n_variants` phrasings, fused by reciprocal rank |
| `hyde` | a hypothetical Wikipedia passage, used as the search key |

`off` is the arm that isolates what rewriting buys over translation alone;
`query_translation.enabled: false` is the harsher baseline that measures what
translation itself is worth. The rewrite must not change what was asked — a
count question that comes back as a *who* question is the failure mode that
disqualified three candidate models
([`configs/models.yaml`](configs/models.yaml)).

---

## Repository layout

```
configs/
  default.yaml        baseline system; every ablation is a diff against this
  models.yaml         provider/model registry with measured capabilities
src/tactistat/
  artifacts.py        atomic writes and artifact manifests
  cli.py              build commands and deterministic stats queries
  config.py           three-layer config: defaults -> experiment -> --set
  data/
    statsbomb.py      fetch and cache event data
    wikipedia.py      scope, resolve, and fetch the text corpus
  stats_tool/
    minutes.py        minutes played, reconstructed from the event stream
    aggregate.py      per-player tables and per-90 normalisation
    query.py          filters -> number + supporting matches
    langchain_tool.py the same engine, bound as a structured tool
  rag_tool/
    chunking.py       section or fixed windows, budgeted in real tokens
    index.py          embeddings + FAISS, with the chunk table beside it
    retrieve.py       dense / bm25 / hybrid, optionally reranked
    langchain_tool.py retrieval bound as a structured tool
  pipeline.py         translate -> route -> tools -> synthesise
  llm/
    registry.py       handle -> chat model, with the measured decoding mode
  query_translation/
    translate.py      any-language question -> English retrieval queries
  router/
    route.py          question -> label + validated tool slots
  synthesis/
    synthesize.py     tool evidence -> cited answer, or an abstention
  eval/               metrics and the ablation harness          (next)
scripts/
  01_build_dataset.py StatsBomb download
  02_build_corpus.py  Wikipedia crawl
  03_build_stats.py   aggregation + minutes reconciliation
  probe_models.py     verify the model registry against live endpoints
tests/                data integrity, resolution, and stats regressions
```

Configuration drives behaviour: an experiment is a config diff, not a code
branch, so any result in the report traces back to the settings that produced
it.

---

## Data

### StatsBomb open data — 2022 FIFA World Cup

64 matches, 234,637 events, complete xG coverage. Not committed (~28 MB);
`scripts/01_build_dataset.py` reproduces it exactly in about 30 seconds.

The dataset is validated against facts sourced from outside it, because a
loading bug produces a table that is the right shape and quietly wrong, and the
evaluation cannot catch that — it grades against ground truth derived from the
same table:

| Check | Result |
| --- | --- |
| Tournament goal total vs. official record | 172 = 172 |
| Per-team goals vs. each match scoreboard | 128 / 128 rows agree |
| Top scorers vs. official record | Mbappé 8, Messi 7, Giroud 4, Álvarez 4 |
| xG coverage | 1,494 / 1,494 shots |

One subtlety is load-bearing: StatsBomb records penalty-**shootout** kicks as
ordinary `Shot` events in `period == 5`. Counting them inflates the tournament
total from 172 to 195 and invents scorers, while every table still looks
normal. `tests/test_data_integrity.py` pins this.

### Minutes played, and why they are not read off the lineups

Every per-90 rate divides by minutes played, so the denominator decides the
answer. Two things make it harder than it looks.

StatsBomb's clock runs through stoppage time — the first half of Iran v United
States ends at 52:04 — and periods therefore *overlap*, because period 2 starts
at 45:00 while period 1 is still running. Raw subtraction across a period
boundary is meaningless, and crediting stoppage time would make a player's
denominator depend on how long the referee added. Each reading is clipped to
its period's nominal end instead, so a full match is 90 minutes and a full
extra-time match is 120.

The lineup `positions` field is the obvious source and is wrong often enough
to matter: reconstructing from it puts *twelve* Iran players on the pitch
simultaneously and disagrees with the substitution record in six matches, by up
to 87 player-minutes. Minutes are therefore rebuilt from the event stream —
`Starting XI`, `Substitution`, `Player Off`/`Player On`, red cards — which
reconciles cleanly:

| Check | Result |
| --- | --- |
| Matches over-counting minutes (impossible if correct) | 0 / 64 |
| Largest shortfall (dismissals, off-pitch spells) | 8.15 min |
| Messi's minutes vs. 5×90 + 2×120 | 689.9 ≈ 690 |
| Goals surviving aggregation vs. official record | 169 + 3 own goals = 172 |

The identity behind the first row is simply that eleven players a side are on
the pitch at all times, so a match must account for `22 ×` its nominal length.
It may fall short; it can never exceed. `tests/test_stats_tool.py` pins it.

### Wikipedia — 318 articles, 4,290 sections

Scope is derived from the event data rather than hand-maintained: the 32
competing teams, 243 players selected by appearances or goals, the tournament's
own pages, and a curated set of tactical concepts.

StatsBomb records legal names while Wikipedia uses common names
(`Lionel Andrés Messi Cuccittini` → `Lionel Messi`), so every entity is
resolved against the live API and each resolution is logged to
`data/processed/wikipedia/resolution_log.json`.

Resolution is verified per entity type, because a wrong article is more
damaging than a missing one — it is well written, on topic, and
indistinguishable from a correct one once indexed. An earlier version accepted
any football-related search hit and pulled in the 2026 and 2030 World Cups
alongside the 2022 one. `tests/test_wikipedia_resolution.py` pins every case.

Wikipedia text is CC BY-SA 4.0; page title, URL, and revision ID are stored
with every record for attribution.

### Retrieval

Chunks are budgeted in the embedding model's **own tokens**, prefix included.
11.5% of this corpus's sections exceed bge-small's 512-token window, and a
sentence-transformer given more than its window truncates and returns an
embedding anyway — no exception, no warning. The tail of every long section
would simply be unreachable, and nothing downstream could say so: retrieval
still returns five passages and recall is just quietly lower. An earlier
chunker split to exactly 512 and *then* prefixed each chunk with its page
title, pushing 70 chunks over; `tests/test_rag_tool.py` pins it.

| Strategy | Chunks | Over the window |
| --- | --- | --- |
| `section`, 512 tokens (baseline) | 5,073 | 0 |
| `fixed`, 300 tokens | 5,893 | 0 |
| `fixed`, 500 tokens | 3,363 | 0 |

Hybrid retrieval fuses **ranks, not scores**. A BM25 score is unbounded and
corpus-relative while a normalised cosine similarity lives in [-1, 1], so a
weighted sum of the two means something different for every query. Reciprocal
rank fusion only asks how highly each retriever placed a document, which is
comparable by construction.

BM25 matches literal terms, so queries and documents are tokenised the same
way — case and accents folded. LangChain's default is `text.split()`, which
left the index case-sensitive: `Lionel Messi` returned his page while
`lionel messi` returned Harry Maguire.

The index carries a manifest folding in the chunking settings, the embedding
model, and a **fingerprint of the corpus text itself** — the page and revision
IDs actually crawled, not just the config that asked for them. Config alone
would let `build-corpus --force` pull newer Wikipedia text while every index
built on it still looked current, citing revisions that no longer say what the
answer claims. The ablation varies exactly these, so a stale index is a likely
accident rather than a hypothetical one.

A rebuild is staged in a scratch directory and swapped in, and the manifest is
removed before any live file is touched. An interrupted rebuild therefore
leaves an index that is *refused* — the config is unchanged, so a surviving
manifest would still match a half-written index.

---

## Roadmap

- [x] Data layer: StatsBomb ingestion, Wikipedia corpus, integrity tests
- [x] Model registry across seven providers, verified against live endpoints
- [x] Stats tool: minutes, aggregation, per-90 normalisation, query interface
- [x] RAG tool: section chunking, embeddings, BM25, hybrid retrieval, reranking
- [x] Query translation, router, cited synthesis, and the `tactistat ask` CLI
- [ ] LangGraph graph and LangSmith tracing over the same stages
- [ ] Evaluation set: 40–50 questions with ground truth
- [ ] Baseline and ablation study (chunking, embedding model, retrieval, rerank)
- [ ] Bootstrap confidence intervals for player comparisons
- [ ] `REPORT.md` — hypotheses, method, results, discussion

---

## License

Code: MIT. Wikipedia content: CC BY-SA 4.0. StatsBomb open data is used under
the terms of the [StatsBomb open data
licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
