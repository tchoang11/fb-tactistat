# Processed data

Artifacts here **are committed**. They are small, and shipping them means a
fresh clone can run retrieval and evaluation immediately — and that everyone
evaluates against the same snapshot rather than whatever Wikipedia looked like
on the day they crawled it.

The large, fully reproducible inputs live in `../raw/` and are gitignored.

| Path | Size | Produced by |
| --- | --- | --- |
| `wikipedia/corpus.jsonl` | ~5 MB | `scripts/02_build_corpus.py` |
| `wikipedia/resolution_log.json` | ~50 KB | same |

## `wikipedia/corpus.jsonl`

One JSON object per line, one line per article.

```json
{
  "page_id": 9899,
  "title": "Lionel Messi",
  "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
  "revision_id": 1368013104,
  "fetched_at": "2026-08-06T16:13:00+00:00",
  "license": "CC BY-SA 4.0",
  "entity_type": "player",
  "entity_key": "Lionel Andrés Messi Cuccittini",
  "query": "Lionel Messi",
  "sections": [
    {"heading": "Introduction", "level": 0, "text": "..."},
    {"heading": "Club career", "level": 2, "text": "..."}
  ]
}
```

`entity_key` is the exact StatsBomb name, which is what joins a page back to
the event data — it is how a claim about a player's page can be tied to a
number computed for that same player.

`revision_id` pins the exact article version behind every citation. Wikipedia
articles change; a citation that only names a page is not reproducible.

`sections` preserves the article's own structure. Baseline chunking splits on
these boundaries rather than a fixed token window, on the hypothesis that a
section is already a coherent semantic unit — the ablation study tests that
against fixed 300- and 500-token chunks.

## `wikipedia/resolution_log.json`

Every entity that was looked up, what it resolved to, and how. Kept because
resolution is the step most likely to be silently wrong: StatsBomb names do not
match Wikipedia titles, so each one is a search, and a search can return a
well-written article about the wrong subject.

```json
{
  "query": "Fred",
  "entity_type": "player",
  "status": "ok",
  "title": "Fred (footballer, born 1993)",
  "detail": "search -> Fred (footballer, born 1993)",
  "sections": 14
}
```

`detail` distinguishes a direct title hit from a search fallback. Search hits
are the ones worth auditing; `grep '"detail": "search' resolution_log.json`
lists them.

## Rebuilding

```bash
python scripts/01_build_dataset.py          # raw/, ~30 s
python scripts/02_build_corpus.py --force   # here, ~90 s
pytest tests/ -q
```

Expect the corpus to differ slightly from the committed snapshot: Wikipedia
articles are edited continuously, so revision IDs and section text move. The
integrity tests assert structure and scope, not exact text.
