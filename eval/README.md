# Evaluation data

`test_set.json` is the versioned, human-readable ground truth. It contains 45
questions: 15 each for `STAT`, `TACTICAL`, and `HYBRID`, with English and
Vietnamese examples in every group.

Numeric labels come from the validated StatsBomb tables. Retrieval labels name
a Wikipedia page and section plus a short phrase from the relevant content,
rather than a chunk id. Chunk ids change with the chunking strategy; content
anchors keep the labels specific and valid across both chunking arms.

`minutes` and `appearances` are deliberately absent. Both are checkable — the
synthesis guard validates numbers claimed of them — but neither is in
`stats_tool.metrics`, so the router has no slot to put them in and every such
question would abstain. Excluding them keeps the set a measurement of the
system rather than of a known gap; promoting them to askable metrics is the
alternative, and it is a code change, not a labelling one.

Every run writes two artifacts. The raw report goes to `eval/results/raw/` and
is ignored by Git: what makes it large is the retrieved passage text and the
rendered tool context, the same Wikipedia paragraphs repeated across items. The
tracked half goes to `eval/results/runs/` — the same file with passage bodies
replaced by their metadata and a character count. Every per-item score, guard
note, judge rationale and passage rank survives, which is what a table in
[`REPORT.md`](../REPORT.md) has to be recheckable against by someone who only
cloned the repository. Per-axis comparisons go to `eval/results/ablations/`,
tracked, each carrying the full effective base config so any arm can be
reconstructed as base plus overrides.

Every run writes one, at roughly 340 KB for the 45-item set. That is a file, not
a commit: track the runs a published table cites and leave the rest local.

The slice cannot be rejudged, and `--rejudge` refuses it by name. Its tool
context is a character count, so grading it would grade the placeholder and
report a number instead of an error. The refusal is on content as well as on a
marker, so a slice written by an older build is caught too; rejudge the raw
report the slice names in `output_path`.

Report paths inside artifacts are written relative to the repository root; an
absolute path names a directory on the machine that produced the run.

Every row records `model_calls` and `cache_hits`. LangChain starts the model
callback before checking its cache, so a cached request still counts as a model
call; `cache_hits` is the field that reveals whether it reached the provider.
The summary reports fully cached items, items with any cache hit, and
`timings_trustworthy`. Even one hit invalidates latency because a cached stage
plus a real stage describes neither a cold nor a warm system. Quality metrics
are unaffected, but a latency table must come from a run with the cache off.

```bash
# Validate schema and labels against committed artifacts; no model or index
tactistat evaluate --validate-only

# Cheap smoke run
tactistat --set tracing.enabled=false evaluate --limit 3 --no-judge

# Full baseline, including the independent answer judge
tactistat --set tracing.enabled=false evaluate

# Latency baseline: no cache, so timings_ms describes the system
tactistat --set llm.cache_enabled=false evaluate --no-judge

# Resume judge failures without paying to run the pipeline again
tactistat evaluate --rejudge eval/results/raw/<run>.json
```

`evaluate` exits non-zero when an item or the judge failed, when a stage could
not run, or when rows are left ungraded, so a broken run cannot be mistaken for
a low-scoring one. A stage that ran and found nothing still scores; a stage that
did not run leaves its metrics out of the denominator and puts the reason in the
row's `stage_errors`.
The first quota error stops further judge requests in that sweep. Rejudging
creates a sibling artifact, keeps the source report untouched, and retries only
rows whose judge result is missing or failed.
When the configured judge model changes, all previous judge scores move to row
history and leave the denominator until that row is graded by the new model, so
one reported mean never mixes judges. Abstentions are deterministic misses
(`correctness=0`, `completeness=0`, `faithfulness=None`) and consume no quota.
