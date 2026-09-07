# TactiStat

An agentic RAG system that answers football questions by routing them to the
right source: computed statistics for numbers, retrieved text for explanations,
and both when a question needs both.

Built on the 2022 FIFA World Cup — 64 matches of StatsBomb event data and 318
Wikipedia articles scoped to the teams, players, and concepts that appear in it.

> **Status: complete and measured.** The data layer, the stats tool, retrieval,
> the full LangGraph pipeline, a labelled 45-question evaluation harness, all six
> ablation axes, and bootstrap confidence intervals are built and tested. Results
> and what they do and do not support are in **[REPORT.md](REPORT.md)**; the
> shortest version is that routing, slot filling, language detection and numeric
> computation score 1.000, retrieval is the bottleneck at Recall@5 0.500, the
> mechanical evidence guard is both stricter and narrower than correctness, and
> a second uncached draw shows the generated queries change completely between
> calls while the retrieval scores barely move.

---

## Why this design

A language model asked *"how many goals did Messi score at the 2022 World Cup?"*
will produce a number. Often the right one, sometimes not, and there is no way
to tell which from the answer alone. Asked *"why did Morocco's press work
against Spain?"*, a SQL-style stats bot has nothing to say at all.

The two question types need different machinery:

| Approach                | Failure mode                                                       |
| ----------------------- | ------------------------------------------------------------------ |
| Retrieval-only RAG      | Numbers get paraphrased out of prose and drift                     |
| Text-to-SQL / stats bot | Correct numbers, no explanation of *why*                         |
| **TactiStat**            | Route by question type, compute numbers, retrieve prose, cite both |

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

# Validate the labelled set without making model calls
tactistat evaluate --validate-only

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

| Provider      | Used for              | Free tier                      | Sign-up                                                                 |
| ------------- | --------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| Groq          | router, synthesis     | ~30 RPM, ~1k req/day per model | [https://console.groq.com/keys](https://console.groq.com/keys)           |
| Google Gemini | translator, LLM judge | flash tier                     | [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

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
evaluation report records the effective handles and config; the CLI also answers
questions end to end and exposes the stats engine deterministically:

```bash
tactistat stats goals Messi
tactistat stats goals Messi Mbappe --per90
tactistat stats goals --top-n 5 --stage "Group Stage"

# End to end, in either language, with every stage's decision shown
tactistat ask "Argentina ghi bao nhiêu bàn?" --trace
tactistat ask "Why was Morocco hard to break down?" --strategy hyde --stream

# Follow-up questions resolve against the conversation
tactistat chat
```

A ranking and a total are different questions, and the router picks between
them: Argentina's top five scorers sum to 14 of the squad's 15 goals, so
answering "how many goals did Argentina score?" from a truncated leaderboard is
wrong by one goal. The router fills an `operation` slot — `player`, `compare`,
`ranking` or `total` — and every slot it returns is validated against the
configured metrics before a tool sees it; each repair is recorded so the
evaluation can report which one fired.

Support is checked per clause, not per answer. "Morocco pressed high [1]. France
won the tournament." cites a passage and is still half invented; so does
"Morocco pressed high [1] and France won the tournament", where the bracket
never leaves the first half of the sentence. Text is split on sentence
punctuation, line breaks, list bullets, `;`, `:` and coordinating conjunctions,
and each clause must carry a citation or restate a number the stats tool
computed — so "Messi scored 7 goals and France won the tournament" does not ride
to safety on the 7.

A bracket covers the text *before* it, up to the previous bracket, which is how
prose actually cites. That single rule separates the two shapes: "pressed high
and defended deep [1]" is covered throughout, "pressed high [1] and France won"
is not. Numbers are checked separately and against the metric they are claimed
of, so "18 goals" is wrong even when 18 appears elsewhere in the evidence as a
date.

What this proves is attribution, not entailment. "France won the tournament [1]"
passes every check above while passage 1 talks about Morocco; a valid citation
rank says a claim was attributed, not that the passage agrees with it. So
`Answer.ok` means the mechanical checks passed, and never means the answer is
faithful — that is a measurement the evaluation takes with a judge on a
different provider and model family, not a boolean the writer awards itself.

The judge deliberately defaults to a different provider *and* model family from
synthesis. LLM judges show a measurable preference for text produced by their
own family, and letting one model both write and grade an answer would put that
bias inside every faithfulness number in the report.

---

## The graph

```
        translate          question -> self-contained English, in one LLM call
            |
          route            STAT / TACTICAL / HYBRID, with validated slots
       +----+----+
    stats     retrieve     both, concurrently, when the question needs both
       +----+----+
        synthesize         cited answer in the asker's language, or an abstention
```

The nodes in [`graph.py`](src/tactistat/graph.py) are thin wrappers over the
stage functions in [`pipeline.py`](src/tactistat/pipeline.py) and hold no logic
of their own, so the routing and evidence contracts live in exactly one place.
The graph also keeps the original question as an intent hint. If a rewrite
drops the prose half of a mixed number-and-explanation question, router
validation restores `HYBRID` and retrieval searches the complete original
request instead of the truncated rewrite.
The graph earns its keep on three things a straight-line function cannot do:

**Both tools at once.** A HYBRID question needs a number and prose, and neither
depends on the other. They share a superstep, so the reported total is wall
clock rather than the sum.

**Conversation memory.** State is checkpointed per `thread_id`, and the
translator sees the previous turns, so an elliptical follow-up resolves before
it reaches the router:

```
> Messi kiến tạo bao nhiêu lần?      -> How many assists did Lionel Messi ...
> Còn Mbappé?                        -> How many assists did Kylian Mbappé ...
```

Memory is opt-in: `pipeline.run(question)` with no `thread_id` is an isolated
run, because an evaluation reuses one pipeline across unrelated questions and a
shared default thread would silently feed each answer into the next. An isolated
run uses a graph with no checkpointer at all, so a 50-question sweep does not
leave 50 dead conversations behind. `chat` names a thread; `ask` does not, and
the store is in-process, so a conversation lasts as long as the command.

A turn is remembered only if it passed the evidence checks *and* every tool the
route asked for returned: a rejected claim would be quoted back as established
context, and a partial answer would be quoted back without the caveat that made
it partial. Earlier turns enter the prompt as the human and assistant messages
they were, never interpolated into the system message — a question typed last
turn must not inherit system authority this turn. The route node also clears
the previous turn's tool results, so a follow-up cannot answer from evidence
that was never retrieved for it.

**Streaming.** `--stream` reports each stage as it finishes.

### Tracing

Every model call is recorded: LangSmith when `LANGSMITH_API_KEY` is set, and a
local `data/traces/trace.jsonl` otherwise, so a clone with no key still runs and
still leaves an audit trail. An ablation pass is roughly 900 traces against a
free tier of 5,000 a month, so `tracing.enabled: false` is the setting for a
large sweep. This exists because temperature 0 is not a reproduction guarantee:
explaining a wrong answer by re-running it is not explaining it.

Every backend is a callback attached to the run, LangSmith included. Switching
LangSmith on the usual way — setting `LANGSMITH_TRACING` in the environment —
would make tracing a process-wide fact, so a second pipeline built with tracing
off would silence the first, and an ablation that traces one arm would trace
them all. Nothing here writes that variable; if the environment already does,
the run warns rather than quietly ignoring the config.

A trace holds whole prompts and answers, so `data/traces/` is git-ignored, and
the test suite unsets the key rather than uploading its stand-in models to a
real project.

---

## Query translation

Questions arrive in Vietnamese or English, often as a fragment — *"Messi bàn
thắng"*. The corpus is English and BM25 matches literal terms, so an
untranslated question retrieves nothing at all. Translation is therefore a
correctness step, not a tuning knob, and it is kept separate from the rewrite
strategy layered on top of it. Both collapse into a single LLM call.

| `query_translation.strategy` | What is retrieved                                                    |
| ---------------------------- | -------------------------------------------------------------------- |
| `off`                        | a literal translation, wording and ambiguity preserved              |
| `rewrite`                    | one self-contained English question                                 |
| `multi_query`                | the question plus `n_variants` phrasings, fused by reciprocal rank |
| `hyde`                       | a hypothetical Wikipedia passage, used as the search key            |

`off` is the arm that isolates what rewriting buys over translation alone;
`query_translation.enabled: false` is the harsher baseline that measures what
translation itself is worth. The rewrite must not change what was asked — a
count question that comes back as a *who* question is the failure mode that
disqualified three candidate models
([`configs/models.yaml`](configs/models.yaml)).

---

## Evaluation

[`eval/test_set.json`](eval/test_set.json) contains 45 labelled questions: 15
each for `STAT`, `TACTICAL`, and `HYBRID`, with English and Vietnamese examples
in every group. Stats values are pinned to the validated processed tables.
Retrieval targets use a Wikipedia page, section, and short content anchor rather
than a chunk id, so labels stay specific without changing when an ablation
changes the chunking strategy.

Each stage is scored separately:

| Stage | Metrics |
| --- | --- |
| Translation | source-language accuracy and degradation count |
| Router | label accuracy, confusion matrix, slot accuracy, items with repairs |
| Stats | exact match with an explicit tolerance |
| Retrieval | Recall@1/3/5 and mean reciprocal rank |
| Synthesis | mechanical evidence validity and abstention count |
| Independent judge | correctness, completeness, and faithfulness on a 0–1 scale |
| Runtime | p50 and p95 latency per graph stage, and whether the cache served them |

There is deliberately no opaque composite score. A wrong route, a missed
passage, and an unsupported answer require different fixes — and they are not
interchangeable: running the same pipeline behind a deliberately worse router
leaves the synthesis guard's score *unchanged* while judged correctness falls
0.878 to 0.722, because a tool asked the wrong question still returns numbers
that check out. The judge uses the
configured `models.judge` role and receives the reference answer for
correctness, but only tool output as evidence for faithfulness.

```bash
# No model or retrieval index needed
tactistat evaluate --validate-only

# Small pipeline smoke run; skip the extra judge call
tactistat --set tracing.enabled=false evaluate --limit 3 --no-judge

# Full baseline with the independent judge
tactistat --set tracing.enabled=false evaluate

# Re-run one labelled case
tactistat --set tracing.enabled=false evaluate --id hybrid-en-05

# Retry only missing/failed judge rows from an existing raw report. This writes
# a new report and never reruns the pipeline stages.
tactistat evaluate --rejudge eval/results/raw/<run>.json
```

Raw reports contain the effective config, test-set fingerprint, stage outputs,
retrieved passages, trace run id, per-item scores, judge rationale, and latency.
They are checkpointed atomically after every item under `eval/results/raw/`, so
a provider failure is recorded without losing the rest of a long run. Summary
metrics are also sliced by expected route and question language.

Each finished run also writes a tracked slice to `eval/results/runs/`: the same
report with the passage bodies and rendered tool context replaced by character
counts, and every score, guard note, judge rationale and passage rank kept. That
is the half a published table has to be recheckable against from a clone, and it
is about 340 KB against the raw report's 900 KB. Both halves record their paths
relative to the repository root. `--rejudge` refuses the slice by name rather
than grading its placeholders, and detects one by content so an older file is
caught too.
If a judge hits a provider quota, the runner circuit-breaks subsequent judge
calls while preserving all deterministic pipeline scores. `--rejudge` can fill
those rows later, or use `--set models.judge=<provider:model>` to grade the same
checkpointed evidence with another configured judge.
An abstention on this answerable labelled set needs no model call: correctness
and completeness are deterministically 0, while faithfulness remains excluded.

End-to-end latency is recorded but not reported as a headline. The LLM cache is
on by default and a hit never reaches a provider, so every row carries
`model_calls` and `cache_hits` and the summary carries `timings_trustworthy`. A
latency figure is only a measurement when it comes from
`--set llm.cache_enabled=false`, and even then the router's own spread — 715 ms
to 32 s on the same prompt — makes p95 on 45 items a sample of about two.
Retrieval latency is the exception, and the ablation reports it, because the
cache never serves local compute.

### Ablation

One axis at a time, each written to `eval/results/ablations/`:

```bash
tactistat ablate retrieval_mode     # dense | bm25 | hybrid
tactistat ablate rerank             # cross-encoder off | on
tactistat ablate chunking           # section boundaries | fixed windows
tactistat ablate translation        # disabled | off | rewrite | multi_query | hyde
tactistat ablate router             # few-shot | keyword baseline
tactistat ablate baseline           # the shipped configuration, end to end

# A second draw of the translator with the cache off, to separate what the
# strategy does from what the provider happened to return. Split in two because
# 120 uncached calls does not fit in the translator's free tier at once.
tactistat ablate translation_uncached          # multi_query | hyde
tactistat ablate translation_uncached_single   # off | rewrite
```

The retrieval axes stop after the retriever and score only Recall@k and MRR;
they never pay for synthesis or a judge. Holding the upstream stages fixed is
what makes an arm comparable to its neighbour, and the LLM cache is what holds
them: within a retrieval axis the translate and route prompts are identical, so
every arm after the first receives exactly the same query. The translation axis
changes that prompt by construction, so nothing is shared there and nothing
should be. Chunking and embedding arms each build their own index, because
sharing one directory would compare an arm against another arm's vectors.

Because the cache serves those upstream stages, their wall clock says how warm
the cache was, not how fast the arm is. Retrieval is the one stage it never
touches, so it is the one stage timed: each arm discards a retrieval to load the
embedding model and the cross-encoder, then reports p50 and p95 over the rest.
That is what lets the rerank axis answer its own question — the cross-encoder
costs 12 ms → 169 ms at p50 for +0.087 MRR.

A failure is recorded, not averaged. The distinction the harness has to make is
between a stage that ran and found nothing and a stage that did not run:

| event | scored |
| --- | --- |
| retriever searched, matched no passage | 0 — a measurement |
| router sent the question away from RAG | 0 — a measurement |
| strategy returned fewer variants than promised | in the mean — that is the strategy |
| index or reranker raised | row error, excluded from metrics *and* latency |
| translator refused by the provider | row error, excluded |

The last two matter because both otherwise read as *good news about speed and
bad news about quality*: an index outage scored as zero recall looks like a weak
retriever that is impressively fast, and a rate-limited translator searches with
the untranslated question, which on a Vietnamese item retrieves almost nothing.
`RagAnswer`, `StatsAnswer` and `Answer` each carry a `failed` flag for exactly
this reason — synthesis treats "nothing found" and "could not run" identically
and is right to; evaluation must not.

The same rule applies end to end, not only in the ablation. Every stage swallows
its own exception so one bad question cannot end a sweep, so
`PipelineResult.stage_errors` names the stages that did not run, and the
evaluation drops exactly those metrics for that row — a broken retriever blanks
Recall and MRR and leaves the router's score alone. Those rows never reach the
judge either, because grading an answer the system could not produce spends
quota to score nothing.

An arm that fails outright leaves its error in the artifact and the sweep runs
the remaining arms. Every count reaches the exit code, so a sweep that lost an
arm — or thirty items — cannot be mistaken for one that finished.

Each artifact embeds the full effective base config, so every arm is
reconstructible as that base plus its own overrides without checking out the
config file the run happened to use.

### Confidence intervals

A World Cup gives a player three to seven matches, and a per-90 rate over seven
matches carries real sampling error that a point estimate hides. A per-90
comparison resamples each player's **matches** — the unit the tournament
actually draws, and the one that keeps a game's minutes and events together —
and reports the interval as evidence:

```
Kylian Mbappé (France): 1.21 [597 min, 7 apps], 95% CI 0.43 to 1.86
Lionel Messi (Argentina): 0.91 [690 min, 7 apps], 95% CI 0.55 to 1.20
difference: 0.29, 95% CI -0.34 to 0.93 (10000 resamples of 13 matches, 1 of
them shared); the interval spans zero, so these matches cannot separate them
```

Two players at the same tournament often meet. Messi and Mbappé shared the
final, so that comparison rests on thirteen fixtures rather than fourteen
player-matches, and the shared match moves both rates at once. The difference
interval is therefore paired: fixtures are stratified into shared, left-only and
right-only, each stratum resampled at its own size, and the shared draw made
once and handed to both players. Two players who never met fall back to
independent draws because their shared stratum is empty.

Pairing is not a way to get narrower intervals; it is a way to get the right
one. It keeps the covariance term, which tightens the interval for two players
who met once in a high-scoring final and *widens* it for two forwards in the
same XI, who tend to split their team's goals rather than score them together.

Only rates get an interval — one around a raw tournament total would describe a
tournament that was not played. The confidence level comes from
`eval.bootstrap.confidence_level` and the rendered label follows it exactly:
0.955 prints as `95.5% CI`, not `96%`.

```bash
# Every interval behind a comparison, with the seed, resample count and the
# match ids each draw was taken over — plus both estimators, paired and not.
tactistat bootstrap goals "Kylian Mbappe" "Lionel Messi" "Julian Alvarez"
```

---

## Repository layout

```
configs/
  default.yaml        baseline system; every ablation is a diff against this
  models.yaml         provider/model registry with measured capabilities
src/tactistat/
  artifacts.py        atomic writes and artifact manifests
  cli.py              build, query, chat, and evaluation commands
  config.py           three-layer config: defaults -> experiment -> --set
  data/
    statsbomb.py      fetch and cache event data
    wikipedia.py      scope, resolve, and fetch the text corpus
  stats_tool/
    minutes.py        minutes played, reconstructed from the event stream
    aggregate.py      per-player tables and per-90 normalisation
    query.py          filters -> number + supporting matches
    bootstrap.py      resampled confidence intervals for per-90 rates
    langchain_tool.py the same engine, bound as a structured tool
  rag_tool/
    chunking.py       section or fixed windows, budgeted in real tokens
    index.py          embeddings + FAISS, with the chunk table beside it
    retrieve.py       dense / bm25 / hybrid, optionally reranked
    langchain_tool.py retrieval bound as a structured tool
  pipeline.py         the stages, and the result they build
  graph.py            LangGraph StateGraph over those stages
  tracing.py          LangSmith when a key exists, a JSONL file otherwise
  llm/
    registry.py       handle -> chat model, with the measured decoding mode
  query_translation/
    translate.py      any-language question -> English retrieval queries
  router/
    route.py          question -> label + validated tool slots
  synthesis/
    synthesize.py     tool evidence -> cited answer, or an abstention
  eval/
    schema.py         strict labelled-set contract
    metrics.py        deterministic per-stage scores
    judge.py          independent answer-quality judge
    runner.py         checkpointed evaluation loop and report
    ablate.py         one axis at a time, arms compared side by side
eval/
  test_set.json       45 labelled English/Vietnamese questions
  results/ablations/  tracked per-axis comparisons, base config embedded
  results/runs/       tracked per-item scores, guard notes, judge rationales
  results/raw/        ignored; the same runs with the passage text in them
scripts/
  01_build_dataset.py StatsBomb download
  02_build_corpus.py  Wikipedia crawl
  03_build_stats.py   aggregation + minutes reconciliation
  probe_models.py     verify the model registry against live endpoints
tests/                data integrity, resolution, stats, pipeline, and evaluation
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

| Check                                     | Result                                   |
| ----------------------------------------- | ---------------------------------------- |
| Tournament goal total vs. official record | 172 = 172                                |
| Per-team goals vs. each match scoreboard  | 128 / 128 rows agree                     |
| Top scorers vs. official record           | Mbappé 8, Messi 7, Giroud 4, Álvarez 4 |
| xG coverage                               | 1,494 / 1,494 shots                      |

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

| Check                                                 | Result                  |
| ----------------------------------------------------- | ----------------------- |
| Matches over-counting minutes (impossible if correct) | 0 / 64                  |
| Largest shortfall (dismissals, off-pitch spells)      | 8.15 min                |
| Messi's minutes vs. 5×90 + 2×120                    | 689.9 ≈ 690            |
| Goals surviving aggregation vs. official record       | 169 + 3 own goals = 172 |

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

| Strategy                           | Chunks | Over the window |
| ---------------------------------- | ------ | --------------- |
| `section`, 512 tokens (baseline) | 5,073  | 0               |
| `fixed`, 300 tokens              | 5,893  | 0               |
| `fixed`, 500 tokens              | 3,363  | 0               |

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

- [X] Data layer: StatsBomb ingestion, Wikipedia corpus, integrity tests
- [X] Model registry across seven providers, verified against live endpoints
- [X] Stats tool: minutes, aggregation, per-90 normalisation, query interface
- [X] RAG tool: section chunking, embeddings, BM25, hybrid retrieval, reranking
- [X] Query translation, router, cited synthesis, and the `tactistat ask` CLI
- [X] LangGraph pipeline: parallel tools, conversation memory, tracing
- [X] Evaluation set: 45 questions, stage metrics, and independent judge
- [X] Ablation harness: one axis per run, arms compared side by side
- [X] Bootstrap confidence intervals for per-90 player comparisons
- [X] [`REPORT.md`](REPORT.md) — method, results, and what they do not support

---

## License

Code: MIT. Wikipedia content: CC BY-SA 4.0. StatsBomb open data is used under
the terms of the [StatsBomb open data
licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
