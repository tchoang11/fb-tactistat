# TactiStat

An agentic RAG system that answers football questions by routing them to the
right source: computed statistics for numbers, retrieved text for explanations,
and both when a question needs both.

Built on the 2022 FIFA World Cup — 64 matches of StatsBomb event data and 318
Wikipedia articles scoped to the teams, players, and concepts that appear in it.

> **Status: in progress.** Data layer and evaluation scaffolding are complete
> (Days 1–2 of the plan). The stats tool, retrieval, router, synthesis, and the
> evaluation study are being built next — see [Roadmap](#roadmap).

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

# Data: ~30 s, writes ~28 MB to data/raw/ (gitignored, fully reproducible)
python scripts/01_build_dataset.py

# Verify the dataset against externally known facts before trusting it
pytest tests/ -q
```

The Wikipedia corpus is **committed to the repository**, so there is no need to
crawl it. To rebuild it anyway (~90 s):

```bash
python scripts/02_build_corpus.py --force
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
| Google Gemini | LLM judge | flash tier | <https://aistudio.google.com/apikey> |

Cerebras, OpenRouter, GitHub Models, OpenAI, and a local Ollama server are also
registered in [`configs/models.yaml`](configs/models.yaml).

---

## Choosing models

Every model is addressed as `provider:alias` and resolved through
[`configs/models.yaml`](configs/models.yaml). Six of the seven providers speak
the OpenAI-compatible wire format and therefore share one client adapter; only
Gemini needs its own. Adding a provider is a YAML edit.

Each role can point at a different model, and any of them can be overridden per
run:

```bash
# Defaults from configs/default.yaml
tactistat ask "How many goals did Messi score at the 2022 World Cup?"

# Override one role
tactistat ask --set models.synthesis=gemini:3.6-flash "..."

# Sweep a role across models as an ablation axis
tactistat eval --sweep models.synthesis=groq:llama-3.3-70b,gemini:3.6-flash
```

The judge deliberately defaults to a different provider *and* model family from
synthesis. LLM judges show a measurable preference for text produced by their
own family, and letting one model both write and grade an answer would put that
bias inside every faithfulness number in the report.

---

## Repository layout

```
configs/
  default.yaml        baseline system; every ablation is a diff against this
  models.yaml         provider/model registry with measured capabilities
src/tactistat/
  config.py           three-layer config: defaults -> experiment -> --set
  data/
    statsbomb.py      fetch and cache event data
    wikipedia.py      scope, resolve, and fetch the text corpus
  stats_tool/         aggregation and per-90 normalisation      (next)
  rag_tool/           chunking, embedding, retrieval, reranking (next)
  router/             question classification                   (next)
  synthesis/          answer generation with citations          (next)
  eval/               metrics and the ablation harness          (next)
scripts/
  01_build_dataset.py StatsBomb download
  02_build_corpus.py  Wikipedia crawl
  probe_models.py     verify the model registry against live endpoints
tests/                data integrity and resolution regressions
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

---

## Roadmap

- [x] Data layer: StatsBomb ingestion, Wikipedia corpus, integrity tests
- [x] Model registry across seven providers, verified against live endpoints
- [ ] Stats tool: aggregation, per-90 normalisation, query interface
- [ ] RAG tool: section chunking, embeddings, BM25, hybrid retrieval, reranking
- [ ] Router and synthesis; end-to-end pipeline
- [ ] Evaluation set: 40–50 questions with ground truth
- [ ] Baseline and ablation study (chunking, embedding model, retrieval, rerank)
- [ ] Bootstrap confidence intervals for player comparisons
- [ ] `REPORT.md` — hypotheses, method, results, discussion

---

## License

Code: MIT. Wikipedia content: CC BY-SA 4.0. StatsBomb open data is used under
the terms of the [StatsBomb open data
licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
