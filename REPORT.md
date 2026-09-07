# Football TactiStat — method and results

An agentic RAG system over the 2022 FIFA World Cup: a router sends a question to
a statistics tool, a retrieval tool, or both, and synthesis writes a cited answer
or refuses. This report states what was measured, how, and what the numbers do
and do not support.

Every table below is read from a committed artifact, named under the table, and
every artifact is in this repository rather than on the machine that produced
it. Each per-axis comparison embeds the full effective configuration it ran
against, so any arm is reconstructible as that base plus its own overrides. The
per-item half — every score, every guard note, every judge rationale, and which
passage came back at which rank — is committed under `eval/results/runs/`. What
is deliberately not committed is the retrieved passage text and the rendered
tool context: bulk rather than evidence, and reproduced by rerunning.

---

## 1. What the system claims to do

Three claims, each testable on its own:

1. **A number in an answer came from a computation, not from the model.** Every
   figure is produced by pandas over the StatsBomb event table and checked back
   against that table before the answer is released.
2. **An explanation came from retrieved text, with a page and section attached.**
3. **The system refuses rather than guesses** when the evidence does not answer
   the question.

The evaluation scores each stage separately. There is deliberately no composite
score: a wrong route, a missed passage, and an unsupported sentence need
different fixes, and averaging them into one number hides which one moved.

---

## 2. Method

### 2.1 The labelled set

45 questions, balanced 15 / 15 / 15 across `STAT`, `TACTICAL`, and `HYBRID`,
with English and Vietnamese items in every group. Numeric labels are pinned to
the validated processed tables. Retrieval labels name a Wikipedia page, a
section, and a short content anchor rather than a chunk id, so a label stays
valid when an ablation changes the chunking strategy.

`tactistat evaluate --validate-only` re-derives every numeric label from the
stats engine and locates every retrieval anchor in the corpus before any model
is called. A label that has drifted fails the run instead of quietly scoring the
system against itself.

The set deliberately omits `minutes` and `appearances`. Both are checkable by
the synthesis guard but neither is an askable metric, so every such question
would abstain — measuring a known gap rather than the system.

### 2.2 What `ok` means

`Answer.ok` reports the **mechanical** evidence checks: every number appears in
the evidence and is claimed of the metric it belongs to, every clause is
attributed, and no citation points at a passage that was never retrieved.

It does not mean the answer is faithful. "France won the tournament [1]" passes
every check above while passage 1 talks about Morocco: a valid citation rank
proves attribution, not entailment. It also does not mean the answer is right:
§5.4 runs the whole pipeline behind a deliberately worse router and the evidence
checks score identically, because a tool asked the wrong question still returns
numbers that check out. Faithfulness is measured separately by a
judge on a different provider **and** a different model family, because LLM
judges show a measurable preference for text from their own family and letting
one model both write and grade would put that bias inside every faithfulness
number here.

### 2.3 Ablation design

One axis at a time, each arm a config diff against the shipped baseline.

The retrieval axes stop after the retriever and score only Recall@k and MRR.
Holding the upstream stages fixed is what makes two arms comparable, and the LLM
cache is what holds them: within a retrieval axis the translate and route
prompts are byte-identical, so every arm after the first receives exactly the
same query. The translation axis changes that prompt by construction, so nothing
is shared there and nothing should be. Chunking arms each build their own index,
because sharing one directory would compare an arm against another arm's
vectors.

Because the cache serves the upstream stages, their wall clock measures how warm
the cache was rather than how fast the arm is. Retrieval is the one stage the
cache never touches, so it is the one stage timed: each arm discards one
retrieval to load the embedding model and the cross-encoder, then times the
rest, and reports p50 and p95 over the 30 items.

A failure is recorded, not averaged, and the distinction that matters is between
a stage that ran and found nothing and a stage that did not run. A retriever
that matched no passage scores 0; a retriever that could not be searched, or a
translator the provider refused, is a row error excluded from the metric *and*
from the latency. Both otherwise read as good news about speed and bad news
about quality — §6.1 is what that looks like when it goes unnoticed. An item
that raises scores nothing rather than zero, and the arm keeps going. An arm that fails outright leaves its error in
the artifact and the remaining arms still run, because a comparison missing an
arm is usable and one that silently lost an arm is not. Both counts reach the
exit code.

---

## 3. Retrieval results

30 of the 45 items carry retrieval ground truth (`TACTICAL` and `HYBRID`).
`top_k = 5`, so Recall@5 is the deepest honest measurement; the harness refuses
a configured depth beyond `top_k` rather than reporting Recall@5 under a
Recall@10 label.

### 3.1 Retrieval mode

| arm | MRR | R@1 | R@3 | R@5 |
| --- | ---: | ---: | ---: | ---: |
| `hybrid` | **0.293** | 0.167 | 0.367 | **0.567** |
| `bm25` | 0.264 | 0.167 | 0.367 | 0.400 |
| `dense` | 0.216 | 0.067 | 0.333 | 0.500 |

`eval/results/ablations/retrieval_mode.json`

Reciprocal-rank fusion beats either retriever alone on both MRR and Recall@5.
The interesting part is the disagreement between the two singles: BM25 has the
better MRR and the worse Recall@5. Lexical matching finds the right section
first when the question shares vocabulary with the article — player and team
names, competition rounds — and finds nothing at all when it does not. Dense
retrieval is the opposite: rarely first, more often somewhere in the five. Fusing
ranks rather than scores is what lets the two contribute without one's unbounded
score scale swamping the other's.

### 3.2 Reranking

| arm | MRR | R@1 | R@3 | R@5 | retrieval p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `on` | **0.303** | 0.167 | 0.400 | **0.567** | 169 ms | 189 ms |
| `off` | 0.216 | 0.067 | 0.333 | 0.500 | 12 ms | 21 ms |

`eval/results/ablations/rerank.json`

The axis asks whether the reranker earns its latency, so the arm measures
latency. Retrieval is local compute — FAISS, BM25, the cross-encoder — that the
LLM cache never serves, which is what makes these two numbers comparable when
the end-to-end timings in §5 are not. Each arm also runs one discarded
retrieval before the clock starts, so 175 ms is steady-state work rather than a
model load charged to the first item.

**+0.087 MRR and +0.067 Recall@5 for 12× the retrieval latency.** Whether that
trades well depends on the budget around it: against a multi-second synthesis
call it disappears, and as a standalone retrieval service it is an order of
magnitude. The absolute numbers wander by a millisecond or two between sweeps
and the ratio does not, which is the part worth quoting. It is not the largest retrieval gain in the study — swapping the
query-translation strategy from `rewrite` to a literal translation is worth
+0.113 MRR (§6) at no measured retrieval-latency cost. Whether it costs anything
end to end is a different question: the translation strategies differ in how
many model calls they make, and this study does not time those.

The mechanism is not reordering. `candidate_k` is 20 against a `top_k` of 5, so
the cross-encoder chooses five out of twenty, and on **all 30 items** its top
five contains at least one passage the un-reranked top five never had. The
recall movement is two-sided to match: four items gain a labelled passage inside
the top five and two lose one, netting the +0.067.

### 3.3 Chunking

| arm | MRR | R@1 | R@3 | R@5 |
| --- | ---: | ---: | ---: | ---: |
| `fixed` | 0.242 | 0.067 | 0.367 | 0.533 |
| `section` | 0.216 | 0.067 | 0.333 | 0.500 |

`eval/results/ablations/chunking.json`

This one goes against the shipped default. Section boundaries were chosen on
principle — a Wikipedia section is a semantic unit and a fixed window is not —
and fixed windows retrieve very slightly better here: one item's worth of
Recall@5 across 30 items.

That is well inside the noise of a 30-item set and is reported as a null result,
not as a reason to switch. The honest reading is that on this corpus the choice
does not matter much, which is itself worth knowing: 11.5% of sections exceed
the embedding model's 512-token window and get sub-split anyway, so the two arms
are less different than their names suggest.

### 3.4 A control the harness passes

`section`, `dense`, `rerank off`, and `rewrite` are the same configuration —
each is the shipped baseline, reached from a different axis. They were run in
four separate sweeps, and all four return MRR 0.216111 and Recall@5 0.500000, to
every digit.

Three of them carry the identical config fingerprint `27217cdc`. `section`
differs (`284e0c67`) in exactly one field, and the artifact names it rather than
asking to be trusted: its `effective_overrides` is
`{"rag.index_dir": "data/index/ablation/section"}` and nothing else. That field
records what an arm *actually* differs by rather than what its axis declared,
which is how the index directory shows up at all — no override mentions it.

That is the control for the study. It says the harness varies only the arm under
test and holds the rest fixed, so a difference between two arms is attributable
to the axis rather than to drift between runs.

It also says exactly what it does not cover. Every arm in §3 receives its query
from the translator through the LLM cache — that is what holds the upstream
stages identical, and it is the reason two arms are comparable at all. It also
means every absolute number in §3 is conditioned on one draw of the translator.
§6.1 measures that draw's variance directly by rerunning with the cache off: the
shipped `rewrite` strategy moves 0.023 MRR between draws. Differences *within*
an axis are exact; the level they sit at is one sample.

---

## 4. Confidence intervals for player comparisons

A World Cup gives a player three to seven matches. "Mbappé 1.21 goals per 90 vs
Messi 0.91" reads like a settled comparison; on seven matches it is mostly
sampling noise. Each player's **matches** are resampled 10,000 times — the unit
the tournament actually draws, and the one that keeps a game's minutes and
events together — and the interval is reported as evidence alongside the point
estimate.

| player | goals / 90 | 95% CI | matches |
| --- | ---: | :---: | ---: |
| Kylian Mbappé | 1.21 | [0.43, 1.86] | 7 |
| Lionel Messi | 0.91 | [0.55, 1.20] | 7 |
| Olivier Giroud | 0.85 | [0.23, 1.45] | 6 |
| Julián Álvarez | 0.77 | [0.16, 1.50] | 7 |

| difference | point | 95% CI | fixtures | separates? |
| --- | ---: | :---: | ---: | :---: |
| Mbappé − Messi | +0.29 | [−0.34, +0.93] | 13 (1 shared) | no |
| Messi − Giroud | +0.06 | [−0.58, +0.73] | 12 (1 shared) | no |
| Mbappé − Álvarez | +0.43 | [−0.41, +1.30] | 13 (1 shared) | no |

`eval/results/bootstrap.json` — every interval above with the seed, resample
count, confidence level and the match ids each draw was taken over. Reproduce
the artifact with `tactistat bootstrap goals <players>`, or one comparison
inside an answer with
`tactistat stats goals "Lionel Messi" "Kylian Mbappe" --per90`.

**No pair among the leading scorers is separable at 95% confidence.** Every
interval spans zero. The ranking that a leaderboard presents as settled is, on
this much data, an ordering the data cannot support — and the system now says so
in the answer rather than reporting the point estimate alone:

> Kylian Mbappé ghi trung bình **1.21** bàn mỗi 90 phút, trong khi Lionel Messi
> ghi **0.91** bàn mỗi 90 phút tại World Cup 2022. Sự chênh lệch trung bình là
> **0.29** bàn mỗi 90 phút (Mbappé cao hơn). Tuy nhiên, khoảng tin cậy 95% cho
> chênh lệch là −0.34 đến 0.93, bao gồm giá trị zero, nên không thể khẳng định
> sự khác biệt này có ý nghĩa thống kê.

### 4.1 The two schedules overlap

Every pair above met in the final. That is one fixture, not two, and it moves
both players' rates at once — the same 120 minutes, the same scoreline — so the
comparison rests on thirteen fixtures rather than fourteen player-matches.

The difference interval is therefore **stratified and partially paired**, and
conditional on how the two schedules actually overlapped. Fixtures are split into
three strata — matches both played, matches only the left player played, matches
only the right player played — and each stratum is resampled with replacement at
its own observed size. The shared draw is made once and handed to both players,
so a shared fixture is in both rates in a given resample or in neither.

The stratum sizes are held fixed, and for these pairs that matters more than the
word "paired" suggests. A pair whose only overlap is the final draws that stratum
1-of-1, so the final is in **every** resample and contributes no variance at all.
Part of the narrowing below is that fixed contribution rather than covariance;
the covariance term is what dominates for the two teammate pairs, where the
overlap is six or seven fixtures.

Holding stratum sizes fixed is what keeps each player's resample the
length of their real schedule; resampling the pooled union instead would let a
player's minutes denominator swing between three matches and eleven, widening
every interval with noise the tournament never had.

| pair | shared | independent | paired | |
| --- | ---: | :---: | :---: | --- |
| Mbappé − Messi | 1 | [−0.56, +1.06] | [−0.34, +0.93] | narrower |
| Messi − Giroud | 1 | [−0.63, +0.79] | [−0.58, +0.73] | narrower |
| Mbappé − Álvarez | 1 | [−0.68, +1.38] | [−0.41, +1.30] | narrower |
| Messi − Álvarez | 7 | [−0.69, +0.86] | [−0.74, +0.96] | **wider** |
| Mbappé − Giroud | 6 | [−0.65, +1.32] | [−0.63, +1.46] | **wider** |

`eval/results/bootstrap.json`, reproduced by
`tactistat bootstrap goals "Kylian Mbappe" "Lionel Messi" "Olivier Giroud" "Julian Alvarez"`.

**Pairing does not mean narrower. It means conditional on what happened.**
Var(L − R) is Var(L) + Var(R) − 2 Cov(L, R), and the paired draw is what lets the
covariance term exist at all. For the three one-shared pairs the interval
tightens, and two effects push that way at once: a positive covariance from a
high-scoring final both contributed to, and that same fixture being held fixed
rather than resampled. The estimator cannot separate them on one shared match,
and this report does not claim it can. Where the two share a whole schedule — Messi and Álvarez played all seven of
Argentina's matches together, Mbappé and Giroud six of France's — the covariance
runs the other way: two forwards in the same XI tend to *split* the goals a team
scores rather than score them together, so a resample that is good for one is
often bad for the other, and the honest interval is wider than pretending they
were independent.

The independent column is not a historical footnote; it is
`difference_interval(..., paired=False)`, still reachable, because a claim that
pairing changed the answer has to be checkable against the estimator it
replaced.

An interval of width zero needs more than a shared schedule: it needs the paired
*difference* to be the same in every fixture. A player against themselves is the
only natural case.

**What "cannot separate them" means.** Mbappé's observed 1.21 per 90 is higher
than Messi's 0.91, and that ordering is arithmetic, not an estimate — it is
exactly what happened. The interval is about the rate those matches support: a
gap spanning zero says these seven matches each do not establish that the
underlying scoring rates differ, so the ordering should not be carried beyond
the tournament that produced it. It does not dispute the tournament's own
numbers.

Only rates get an interval. One around a raw tournament total would describe a
tournament that was not played.

---

## 5. End-to-end baseline

All 45 items, shipped configuration, no pipeline errors.

| stage | metric | score | n |
| --- | --- | ---: | ---: |
| Translation | source-language accuracy | **1.000** | 45 |
| Router | label accuracy | **1.000** | 45 |
| Router | slot accuracy | **1.000** | 45 |
| Stats | exact match | **1.000** | 30 |
| Retrieval | Recall@5 | 0.500 | 30 |
| Retrieval | MRR | 0.216 | 30 |
| Synthesis | evidence checks passed | 0.689 | 45 |
| Judge | correctness | 0.878 | 45 |
| Judge | completeness | 0.750 | 45 |
| Judge | faithfulness | 0.927 | 41 |
| — | answered and passed every check | 0.600 | 45 |

`eval/results/ablations/baseline.json`, per-item in `eval/results/runs/`. Judge:
`dashscope:qwen-plus` — see §5.3.
Faithfulness has n = 41 because the four abstentions assert nothing, and a judge
scores an assertion-free answer perfectly faithful; averaging that in would let a
system that refuses everything report faithfulness 1.0.

### 5.1 The deterministic stages are solved; retrieval is the bottleneck

Routing, slot filling, language detection, and numeric computation are at 1.000.
Not one of the 15 `STAT` items failed, and every one of the 30 stats values
matched its label exactly. Two items needed a router repair, and the repair held.

Every failure is downstream of a retrieved passage. Recall@5 is 0.500: half the
labelled sections never reach the synthesiser, and the four abstentions are the
system reporting that honestly rather than filling the gap. §3 is where the
remaining work is, and §3.2 says a reranker is the cheapest part of it.

### 5.2 The guard is stricter than correctness, in a specific way

The evidence checks pass 0.689 of answers while the independent judge scores
correctness at 0.878. Splitting the non-abstaining items by what the guard
decided (the judge never sees that decision):

| | n | judge correctness | judge faithfulness |
| --- | ---: | ---: | ---: |
| guard passed | 27 | 0.981 | 0.944 |
| guard rejected | 14 | 0.929 | 0.893 |

`eval/results/runs/20260819T105517235416Z-0dfd388f.json`. The split is
`result.answer.ok` against `scores.judge_*` over the 41 non-abstaining rows; the
three categories below are `result.answer.note` on the 14 rejected ones.

The guard is informative — passing correlates with a higher independent score —
but it rejects answers that are 93% correct. The 14 rejections break down as:

- **8 — a number written as a word.** "Messi scored seven goals" where the tool
  computed 7. The fact is right; the rule says copy the tool's digits. This is
  deliberate — spelling a number out is exactly how a wrong figure would be
  laundered past a digit-for-digit check — but it means roughly a fifth of the
  set fails on orthography rather than on evidence. It also cascades: a sentence
  whose only number is spelled out is not recognised as a stats restatement, so
  it collects a second, misleading "uncited claim" note for the same root cause.
- **5 — genuinely uncited prose.** A sentence of explanation with no bracket.
  This is the class the guard exists for, and it is 11% of the set, not 40%.
- **1 — a number claimed of a metric the tool never computed.**

The 45 items decompose without overlap. 31 pass the mechanical checks — 27
answered and passed, plus the 4 abstentions, which assert nothing and so pass by
construction — and 14 are rejected: 8 for spelling a correct number as a word, 5
for genuinely uncited prose, 1 for a metric the tool never computed.

`pipeline_success` is 0.600 rather than 0.689 because it requires an answer as
well as valid evidence, so it counts the 4 abstentions as misses. Neither number
is "the score": 27/45 answered cleanly, 4/45 refused, 8/45 were right and
formatted wrong, 6/45 were unsupported or misattributed.

### 5.3 The configured judge cannot finish one evaluation

The shipped `models.judge` is `gemini:3.6-flash`. On its free tier it graded six
items and then returned `RESOURCE_EXHAUSTED`; the runner's circuit breaker
skipped the remaining 39 rather than burning the quota further, leaving every
deterministic score intact and `judge_correctness` on a denominator of six.

A mean over six items is not a result, and the run said so: the CLI printed
`judge errors: 39` and exited non-zero. The table above comes from re-grading
the same checkpointed evidence with `dashscope:qwen-plus` — a different provider
and a different model family from synthesis, so the independence constraint in
§2.2 still holds — via `evaluate --rejudge`, which never re-runs the pipeline.

The default is left as it is rather than quietly swapped, because "which judge"
is a measurement decision and not a bug fix. But a default that cannot complete
one run of the evaluation it is configured for is a real defect, and it is only
visible because the run refused to average six items into a number.

---

### 5.4 The router, and what the guard cannot see

The keyword baseline is the same pipeline with `router.strategy: keyword` — a
regex-and-vocabulary decision instead of a few-shot model call. Both arms, all
45 items:

| metric | `few_shot` | `keyword` | n |
| --- | ---: | ---: | ---: |
| route label accuracy | **1.000** | 0.822 | 45 |
| route slot accuracy | **1.000** | 0.711 | 45 |
| stats exact match | **1.000** | 0.733 | 30 |
| retrieval Recall@5 | **0.500** | 0.433 | 30 |
| judge correctness | **0.878** | 0.722 | 45 |
| judge completeness | **0.750** | 0.611 | 45 |
| judge faithfulness | 0.927 | 0.926 | 41 / 34 |
| **answer evidence checks** | **0.689** | **0.689** | 45 |
| abstentions | 4 | 11 | 45 |

`eval/results/ablations/router.json`, per-item in `eval/results/runs/` —
`…105455541462Z` for `few_shot` and `…105504261464Z` for `keyword`.

The few-shot router earns its model call: 8 of 45 questions get the wrong label
without it and 13 get the wrong slots, and that propagates — 8 of 30 stats
values come out wrong, and Recall@5 falls because a misrouted `HYBRID` item
searches with a worse query.

**Two rows of that table are the interesting ones.** The evidence checks score
*exactly the same* under both routers, and faithfulness moves by 0.001 — while
correctness falls by 0.156.

That is not noise; it is the shape of the guarantee. The guard verifies that
every number in the answer appears in the tool's output and is claimed of the
metric it belongs to. Under the keyword router the tool was asked the wrong
question and answered it correctly, so every number checks out. The judge asks
whether the answer is right, and the judge is the only thing here that noticed.
Faithfulness behaves the same way for the same reason: an answer that faithfully
reports the wrong computation is still faithful to it.

So `answer_evidence_valid` bounds one failure mode — a number the system did not
compute — and says nothing about a number it computed for the wrong question.
§5.2 showed the guard is stricter than correctness; this shows it is also
narrower, and the two are separate facts about the same number.

**Refusal absorbs most of the damage.** Abstentions rise from 4 to 11: of the 8
wrong stats values, 5 produced no usable result and the system declined rather
than answering. Correctness scores those 0 — a refusal on an answerable question
is a miss — but it is the miss the design prefers. Of the 3 that did answer, the
guard caught 2; the one it passed asked for a top-3 leaderboard and returned a
top-5, which the judge still scored correct.

---

## 6. Query translation

The corpus and the embedding model are English-only and BM25 matches literal
terms, so a Vietnamese question retrieves nothing untranslated. Translation is
therefore a correctness step, and the strategy layered on top is the ablation.

| arm | MRR | R@1 | R@3 | R@5 | retrieval p50 | degraded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` (literal) | **0.329** | **0.233** | 0.344 | 0.461 | 10 ms | 0 |
| `hyde` | 0.328 | 0.200 | **0.383** | **0.567** | 15 ms | 0 |
| `multi_query` | 0.264 | 0.133 | **0.383** | 0.467 | 39 ms | 0 |
| `rewrite` (shipped) | 0.216 | 0.067 | 0.333 | 0.500 | 13 ms | 0 |
| `disabled` | 0.190 | 0.100 | 0.211 | 0.328 | 15 ms | 0 |

Every arm here searched with one query except `multi_query`, so the 10–15 ms
spread in that column is noise and the 39 ms is not.

`eval/results/ablations/translation.json`

**Translation earns its place — but not the shipped strategy.** The `disabled`
arm searches with the raw question and is worst on MRR, R@3 and R@5; it is the
only arm that sends Vietnamese text at an English index. Every strategy that
actually translates beats it here.

That result holds on a second, uncached draw against the live provider: every
translating arm beats `disabled` there too, `rewrite` included (§6.1). An earlier
draw put `rewrite` below it, and this section previously said so; one draw did
not support the claim and the correction is in §6.1.

**`off` translates literally and `rewrite` expands** — on the same Vietnamese
item:

```
disabled     High press trong bóng đá hoạt động như thế nào?
off          How does high press work in football?
rewrite      How does high pressing work in football at the 2022 FIFA World Cup?
```

The expansion adds "at the 2022 FIFA World Cup", which every article in this
corpus is about. It broadens the query without adding information, which is
exactly the trade that helps recall a little and hurts rank a lot: `rewrite`
holds a respectable R@5 of 0.500 and finds the right section *first* on 2 items
out of 30, against 7 for `off` and 3 for sending the untranslated question.

**More queries did not buy more.** `multi_query` fuses four phrasings, costs
three times the retrieval time of any single-query arm (39 ms p50, four searches
plus fusion), and lands below a single literal translation on every column but
R@3. Its variants drift toward the general — "how teams apply pressure high up
the pitch in World Cup matches" — while the labelled passage is specific, and
rank fusion then rewards whatever all four happen to agree on.

It is also the only arm in the study whose score is not reproducible: §6.1
measures it moving ±0.05 MRR between draws while every other arm stays inside
±0.01. Four generated queries are four chances to drift and fusion is sensitive
to all of them, so its 0.048 lead over `rewrite` in the table above is inside its
own noise and is not a result.

**HyDE reaches the best recall in the study, and so do two other arms.** Its
R@5 of 0.567 is exactly the reranker's and exactly hybrid retrieval's. Three
unrelated mechanisms — generating a hypothetical passage, reranking twenty
candidates, fusing two retrievers — stop at the same number. On a set this size
that reads as a ceiling rather than a win, and the ceiling is the more useful
observation: roughly two in five labelled passages are not reachable by any of
them. HyDE gets there at single-query retrieval cost, because the hypothetical
passage is generated upstream and the search itself is one query.

Taken together: keep translation, and `rewrite` is the weakest strategy that
still translates. It is last on MRR among translating arms and finds the right
section first on 2 items of 30 against `off`'s 7.

It is not last on everything, and the exception is the reason no default moves
here on this evidence: `rewrite` has the better Recall@5, 0.500 against `off`'s
0.461. Ranking first more often and appearing at all more often are different
properties, and which one matters depends on whether synthesis reads rank 1 or
all five — it reads all five. So `off` is the candidate, not the conclusion, and
the honest recommendation is to run both against a larger set rather than to
swap a default on 30 items where the two arms disagree about which is better.

### 6.1 What is reproducible here, and what is not

Two different questions hide behind the word "reproducible", and this axis
separates them.

**The harness does not drift.** §3.4 is the control: four arms that are the same
configuration reached from four different sweeps return MRR 0.216111 and
Recall@5 0.500000, to six decimals. Rerunning any retrieval axis reproduces it
exactly.

**The generated text does drift, completely.** The sweeps above run against the
LLM cache, which is what holds the upstream stages fixed between arms and also
what hides that the model does not return the same thing twice. Rerunning with
`llm.cache_enabled: false` asks the provider again. At temperature 0, on
`multi_query`, **28 of 28 items came back with different query variants** and
only 6 kept even the same primary query:

```
cached    'pressing tactics in the 2022 FIFA World Cup'
          'tactical pressing styles used by teams in Qatar 2022'
live      'pressing tactics in association football'
          'how teams apply high press in matches'
```

**The scores mostly do not.** Turning the cache off re-runs the *whole* upstream
— the translator and the LLM router — so the honest comparison is on rows where
both draws still routed the question to retrieval. Every cached arm routed all
30 items to RAG; the live draws sent 1 to 3 items elsewhere, which is the router
changing its mind, not the strategy retrieving worse.

| arm | cached | live | change | items |
| --- | ---: | ---: | ---: | ---: |
| `off` | 0.389 | 0.389 | **0.000** | 21 |
| `hyde` | 0.247 | 0.249 | +0.002 | 21 |
| `rewrite` | 0.248 | 0.256 | +0.008 | 20 |
| `multi_query` | 0.271 | 0.321 | **+0.050** | 28 |

`eval/results/ablations/translation_uncached.json` and
`translation_uncached_single.json`, restricted to items both draws retrieved for.
`disabled` is 0.190 in every sweep because it makes no model call at all.

So the queries churn completely and the retrieval score barely moves — except
for the one arm that fuses four of them. `multi_query` is the only arm outside
±0.01, and an earlier draw moved it −0.047 where this one moves +0.050, so its
main-table figure carries roughly ±0.05 that the other arms do not. That is
larger than its 0.048 lead over `rewrite`, and the lead is therefore not a
finding. The `off`-over-`rewrite` gap of 0.113 is twice the largest swing
observed and survives.

**A retraction.** An earlier draw of this axis put `rewrite` at MRR 0.170 against
`disabled`'s 0.194 on the same items, and an earlier version of this section drew
the obvious conclusion: that the shipped strategy was not worth its model call.
This draw does not reproduce it — `rewrite` scores 0.223 against 0.197 — and
every translating arm beats no translation in both draws. The claim was one
sample presented as a result, and the correction is the same lesson as §6's own
recommendation: nothing here moves a default on 30 items and one draw.

**The refusals are the other half.** The live draws had 6 to 8 of 30 translator
calls refused with `RESOURCE_EXHAUSTED`. The harness records those as row errors
and drops them from the denominator, so `off` reports MRR over n=22 and the CLI
exits non-zero. Under the previous behaviour they would have been scored: a
failed translation falls back to the raw question, which on a Vietnamese item
retrieves almost nothing, so the same artifact read the old way reports **`hyde`
at 0.174 over n=30** — below the arm that does not translate at all. That is not
hypothetical; it is this run, scored the way the harness used to score it, and it
is what the first version of this section reported as a property of HyDE.

A sweep long enough to be interesting is long enough to hit a quota, and a
harness that cannot tell the system's behaviour from the provider's will publish
the outage as a result.

**What this does not measure.** The cache is process-global, so it cannot be
switched off for the translator alone: these numbers bound the variance of
translation *and* routing together, and the routed-to-RAG restriction removes the
router's most visible effect but not its influence on the query text. Isolating
the translator needs a deterministic router, which would change the recall levels
and make the arms incomparable with the main table. Nothing here measures how
much synthesis varies the same way.

---

## 7. What this study does not establish

**Retrieval numbers rest on 30 items.** A one-item change moves Recall@5 by
0.033. Differences smaller than that — the chunking axis in particular — are
reported as null results, and no arm was promoted to the default on the strength
of one.

**And on one draw of the translator.** §6.1 measures the second draw: arms move
between +0.007 and −0.047 MRR when the provider is asked again at temperature 0.
The comparisons inside an axis are exact because every arm shares one cached
draw, but any absolute level in §3 carries that band, and any gap narrower than
about 0.05 MRR between two *translating* arms is not established by this study.
Nothing here measures how much the router or synthesis models vary the same way;
the axis that would show it has not been run.

**Attribution is not entailment.** The mechanical checks prove every claim points
at evidence; only the judge asks whether the evidence agrees, and the judge is
itself a language model scoring on a four-point scale — one model, one pass, no
human agreement study behind it. The judge numbers in §5 rank the arms; they do
not certify the answers.

**The headline number depends on a formatting rule.** A fifth of the set fails
because a correct figure was spelled out as a word. Relaxing that rule would move
`pipeline_success` from 0.600 to roughly 0.78 without changing a single answer,
which is the clearest evidence in this report that a single aggregate is the
wrong thing to quote. §5.2 gives the decomposition instead.

**End-to-end latency is not reported.** It is recorded, but the LLM cache is on
by default and a cache hit never reaches a provider, so every row carries
`model_calls` and `cache_hits` and the summary carries `timings_trustworthy`.
Even with the cache off, the router returned in 715 ms and 32 s on the same
prompt in one sitting; p95 over 45 items is a sample of about two, and reporting
it would be reporting one draw's tail as the system's.

**Retrieval latency is reported, and is one laptop's.** §3.2 compares two arms
on local compute the cache never serves, which is what makes the comparison
meaningful at all. It is still 30 queries in one process on one machine with no
GPU — the ratio between the arms is the finding; neither absolute number
transfers, and the cross-encoder is exactly the component a GPU would change
most.

**The committed artifacts hold the scores, not the corpus.** `eval/results/runs/`
carries every per-item score, guard note, judge rationale and passage rank, which
is enough to recheck every table here. Rechecking whether a *labelled* passage
really says what the label claims needs the passage text, and that means
rebuilding the corpus with `scripts/02_build_corpus.py`.

**The bootstrap assumes fixtures are exchangeable, and a knockout is not.** A
player reaches the final only by winning the semi, so resampling a World Cup
schedule with replacement builds tournaments that could not have happened — six
finals, no group stage. The intervals describe sampling variation in the rate
*given these matches*, which is the right question for "is this gap larger than
noise", and they are not a forecast of a replayed tournament. Seven matches is
also few enough that a percentile interval is coarse: the bounds move on whole
goals, not smoothly.

**One provider carries two stages.** The router and synthesis both run on Groq,
so a bad moment there is correlated across two of the five stages rather than
independent.

**The set is thin where the pipeline is weakest.** 19 of 30 stats items ask for
`goals`; there is one `per90` item, three `ranking`, three `compare`, and one
`opponent`. Those are the slots the router repairs most often, and three
`compare` items means one wrong answer moves that cell by 33 points. Widening
the set is the first thing a next iteration should do — and it changes the
test-set fingerprint, so it invalidates comparison with everything above.

---

## 8. Reproducing

```bash
tactistat evaluate --validate-only          # labels vs committed artifacts
tactistat ablate retrieval_mode             # section 3.1
tactistat ablate rerank                     # section 3.2
tactistat ablate chunking                   # section 3.3
tactistat ablate translation                # section 6

# Section 4: writes eval/results/bootstrap.json with both estimators, the seed,
# the resample count, and the match ids behind every interval
tactistat bootstrap goals "Kylian Mbappe" "Lionel Messi" "Olivier Giroud" \
  "Julian Alvarez"

# The same comparison as an answer
tactistat stats goals "Lionel Messi" "Kylian Mbappe" --per90

# Sections 5 and 5.4 reach synthesis and a judge. They used this judge, because
# the shipped default exhausts its free quota partway through (section 5.3).
tactistat --set tracing.enabled=false --set models.judge=dashscope:qwen-plus \
  ablate baseline
tactistat --set tracing.enabled=false --set models.judge=dashscope:qwen-plus \
  ablate router
```

Every run writes its effective config and the test-set fingerprint into its own
report, so a table here can be checked against the settings that produced it —
and the per-axis artifacts embed the whole base config, so an arm is
reconstructible without knowing which config file was checked out.

Numbers here were produced with the LLM cache on, which is correct for quality
metrics and wrong for latency; every row carries `model_calls` and `cache_hits`
and each summary carries `timings_trustworthy` so the difference is visible
rather than assumed. The one latency table in this report (§3.2) measures local
retrieval, which the cache never serves.
