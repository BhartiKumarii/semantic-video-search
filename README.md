# Semantic Video Search

Finds the exact moment in a recording where something was said, by meaning, not keywords.

Ask a question in plain language ("what method avoids recomputing attention over all
previous tokens?") and get back the transcript segments that answer it. Each result
includes the recording, a timestamp and the speaker, even when the words in the
transcript are different from the words in the question.

## How it works

```
video ──► transcript + slide OCR ──► segments in Postgres ──► embeddings ──► pgvector (HNSW) ──► FastAPI search
          (tools/)                   (starter/load_corpus.py)  (all-MiniLM-L6-v2,              (apps/bharti/main.py)
                                                                384 dims, normalized)
```

1. **Extraction (`tools/`).** Whisper (transcription) and sherpa-onnx (speaker
   diarization) turn each recording into a diarized,
   timestamped transcript (`*.diarized.srt`). Frames are sampled at 1 fps and run
   through Tesseract OCR (`ocr.json`). YOLO object and face detection are also
   available.
2. **Loading (`starter/load_corpus.py`).** The transcript is split into segments.
   Each segment gets the slide text that overlaps it in time, then goes into a
   `segments` table with a generated full-text (`tsvector`) column.
3. **Embedding.** Each segment's transcript text is embedded with
   `sentence-transformers/all-MiniLM-L6-v2` (384 dims, normalized, so cosine
   similarity is a dot product) and stored in a `vector(384)` column.
4. **Indexing.** pgvector HNSW index with cosine ops (`m=16, ef_construction=64`).
5. **Search (`apps/bharti/main.py`).** A FastAPI service embeds the query with the
   same model and returns the nearest segments (`ORDER BY embedding <=> query`).
   Keyword (Postgres full-text) and hybrid (Reciprocal Rank Fusion of vector +
   keyword) are also available, so you can compare methods. Vector is the default.

## Results

All numbers come from the golden set in `starter/golden.jsonl`: 17 questions over two
recordings. Each question has the time window that contains its answer. A result counts
as a hit if it is from the same recording and its time span overlaps that window (the
same rule `starter/eval.py` uses). Settings: `all-MiniLM-L6-v2`, k = 5.

| Method  | Recall@5 | MRR  |
|---------|---------:|-----:|
| Keyword (Postgres full-text) | 0.18 | 0.15 |
| **Vector (pgvector)**        | **0.94** | **0.65** |
| Hybrid (RRF, vector + keyword) | 0.82 | 0.64 |

**Key finding: the simplest setup wins.** Pure vector search over cue-level transcript
segments beat hybrid search, and every "fancier" setup I measured did worse:

- **Hybrid regressed** (0.82 vs 0.94 recall). Keyword search is too weak on this
  corpus (0.18), and RRF gives weight to its confident-but-wrong ranks, pushing good
  vector hits out of the top 5. Adding any weight to the keyword list (`w_kw > 0`)
  lost recall.
- **Fusing slide OCR into the embedded text regressed** (−0.18 recall). Slide
  boilerplate adds noise.
- **Merging cues into larger (~350 char) chunks regressed** (−0.23 recall). One
  vector ends up averaging several ideas.
- **HNSW `ef_search` (10 / 40 / 100) made no difference.** At roughly 450 vectors the
  index already returns the exact nearest neighbours.
- **`bge-small-en-v1.5` tied MiniLM on recall** (0.94). It had better MRR but about
  2.5× the query latency, so MiniLM is the default.

The one question vector search misses is a metaphor (a "flavored water" analogy). It
shares no words or meaning with the question, and none of the variants above fixed it.

These findings are recorded in the docstring and comments of
[`apps/bharti/experiments.py`](apps/bharti/experiments.py). Each experiment changes
one thing at a time, so you can trace every difference to that one change.

## Running it

### Requirements

- Python 3.11 or 3.12
- Postgres with the [pgvector](https://github.com/pgvector/pgvector) extension
- Python packages: `apps/bharti/requirements.txt` (FastAPI, uvicorn, psycopg,
  pgvector, sentence-transformers, numpy)
- Rebuilding the corpus from raw video is optional, because `corpus/` is already
  extracted. If you do want to rebuild it, see [`tools/README.md`](tools/README.md)
  (ffmpeg, whisper.cpp, Tesseract).

### 1. Postgres + pgvector

```bash
docker run -d --name pgv -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
psql "postgresql://postgres:postgres@localhost/postgres" -c "CREATE DATABASE semantic_search_bharti;"
psql "postgresql://postgres:postgres@localhost/semantic_search_bharti" -f starter/schema.sql
```

### 2. Environment variables

Everything reads the standard Postgres variables:

| Variable | Default |
|----------|---------|
| `PGHOST` | `localhost` |
| `PGPORT` | `5432` |
| `PGDATABASE` | `semantic_search_bharti` in `apps/bharti/`, `semantic_search` in `starter/` |
| `PGUSER` | `postgres` |
| `PGPASSWORD` | `""` (`main.py`, `starter/`), `postgres` (`evallib.py`) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` (`main.py` only; must match the model used to embed the corpus) |

The defaults differ between folders, so set `PGDATABASE` explicitly:

```bash
export PGHOST=localhost PGUSER=postgres PGPASSWORD=postgres PGDATABASE=semantic_search_bharti
```

### 3. Install, load, embed, index

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r apps/bharti/requirements.txt

cd starter
python load_corpus.py ../corpus/day1 day1
python load_corpus.py ../corpus/day2 day2
python embed.py          # fills the embedding column
python eval.py           # Recall@5 / MRR / latency for keyword, vector, hybrid
cd ..

psql -c "CREATE INDEX ON segments USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
```

### 4. Run the search app

```bash
cd apps/bharti
uvicorn main:app --host 127.0.0.1 --port 8000
```

- `GET /`: a minimal search page with a method picker (vector / keyword / hybrid)
- `POST /api/search` with `{"query": "...", "k": 5, "method": "vector"}`: JSON hits
  with `id`, `video`, `start_s`, `end_s`, `speaker`, `text`, `score`
- `GET /healthz`: liveness check

```bash
curl -s localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"query": "what does cosine similarity measure?", "k": 5}'
```

### 5. Run the experiments

```bash
cd apps/bharti
python experiments.py explore    # nearest neighbours for a few golden questions (in memory, no DB)
python experiments.py pgvector   # keyword vs vector + HNSW ef_search sweep
python experiments.py models     # MiniLM vs bge-small-en-v1.5
python experiments.py chunk      # cue-level vs merged chunks, with/without OCR fusion
python experiments.py hybrid     # keyword / vector / hybrid scorecard + keyword-weight sweep
```

- `pgvector`, `models` and `chunk` build their own temporary tables in the
  database.
- `hybrid` scores the real `segments` table, so complete step 3 first.
- `evallib.py` sets `HF_HUB_OFFLINE=1`, so the models must already be in your
  local Hugging Face cache. `models` needs to download `bge-small-en-v1.5`; to
  allow that, run it with `HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`.

## Repository layout

```
apps/bharti/   main.py (search service), evallib.py (shared eval toolkit), experiments.py
starter/       schema, corpus loader, embed/search modules, scorer, golden set
tools/         video → transcript / frames / OCR / object / face extraction
corpus/        extracted transcripts and JSON for two recordings (day1, day2)
```

## Credits

Built on a course-provided extraction pipeline, corpus and evaluation harness
(`starter/`, `tools/`, `corpus/`). The embeddings, indexing, experiments and search app
in `apps/bharti/` are my own work.
