# securiport-proj1

## Setup

```bash
conda env create -f environment.yml
conda activate securiport-proj1
```

## Directory Setup

```
data/
  archive/
  curated/
    curated_urls.json
  synthetic/
    passengers/
      synthetic_negative_001.json
  runs/
    <run_id>/
      input.json
      sources.json
      chunks.json
      evidence.json
      aggregation.json
      final_report.json
docs/
  check_curated_scraping.md
  evaluate_curated_runs.md
scripts/
  check_curated_scraping.py
  evaluate_curated_runs.py
src/
  api/
  collection/
  frontend/
  nlp/
  processing/
test/
```

data/
- stores local data, curated/ holds manual urls and synthetic/ is for our made up data

data/runs
- store generated outputs from running the pipeline

data/archive
- store older runs for archival purposes, put them here so they don't bloat up the /runs folder

src/collection/
- web scraping stack, handle search engine API calls, URL filtering, HTML text extraction

src/processing/
- data prep stage, filter sources by relevance, chunking logic, filter chunks by relevance

src/nlp/
- analysis stack, evidence snippet extraction logic (using sentiment classifier or LLM), aggregation, summarize evidence with LLM for final report

src/api/
- API layer

src/frontend/
- simple frontend webapp for improved usability

tests/
- unit tests

## JSON Schemas
check schema.json

## Testing Scripts
- validate_run_artifacts.py: tests if pipeline correctly created all run artifacts, provides basic run summary for report
```
python scripts/validate_run_artifacts.py
```

## Model Hosting

Create an .env file in root to setup OLLAMA. Just go into their website, create an account and get a free api key. I'm using gpt-oss:20b for the model.

```
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_key_here
OLLAMA_EVIDENCE_MODEL=gpt-oss:20b
OLLAMA_REPORT_MODEL=gpt-oss:20b
```

All available cloud models Ollama provides can be found on https://ollama.com/search?c=cloud, or with this command in powershell:
```
(Invoke-RestMethod "https://ollama.com/api/tags").models | Select-Object name, size, modified_at
```

Note: Ollama free tier usage limits is pretty generous. Ran evidence extraction 3 times, used like 0.6% of weekly limit (usage limits resets every week).

## Search API (for `/sources/from-search-api`)

Uses Tavily to find public URLs for a target name (Bing's public Search API was retired, so we're not using that despite what the schema example used to say). Add to `.env`:
```
TAVILY_API_KEY=your_key_here
```
Free tier: 1,000 searches/month. Get a key at https://tavily.com.

## Streamlit Dashboard

A simple Streamlit frontend is available for running the full pipeline and viewing generated run artifacts without using Swagger or manually opening files from `data/runs`.

In a second terminal, run the Streamlit dashboard from the project root:

```bash
streamlit run src/frontend/app.py
```

Open the Streamlit app in the browser. By default, Streamlit usually runs at:

```bash
http://localhost:8501
```

The dashboard assumes the FastAPI backend is running at:
```bash
http://127.0.0.1:8000
```

## FastAPI Endpoints

Run the API server from the project root:

```bash
uvicorn src.api.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
GET /health
```

### Source Creation

Creates `input.json` and `sources.json`.

`POST /sources/from-synthetic`

Loads a synthetic case file from `data/synthetic/passengers/`.

`POST /sources/from-curated-urls`

Loads curated URLs from `data/curated/curated_urls.json` for a `target_name`, then scrapes/extracts source text. Use this for a fixed, hand-picked source list.

`POST /sources/from-search-api`

Runs a Tavily web search for `target_name` (query defaults to `"<target_name>" news` if `search_query` is omitted), takes the top `max_urls` results (default 3), then scrapes/extracts source text the same way as the curated-urls flow. Use this when you don't want to hand-pick URLs ahead of time.

### Chunking / Processing

Reads `sources.json` and creates `chunks.json`.

`POST /processing/chunk-llm`

Implemented. Uses paragraph-aware chunking for the LLM pipeline.

`POST /processing/chunk-classifier`

Later endpoint. Will use sentence-window chunking for the sentiment classifier pipeline.

### Evidence Extraction

Reads `chunks.json` and creates `evidence.json`.

`POST /nlp/evidence-llm`

Uses an LLM to extract exact evidence quotes and quote-level sentiment labels.

`POST /nlp/evidence-classifier`

Later endpoint. Uses a sentiment classifier to label sentence/window chunks. Might skip entirely tbh

### Aggregation

Reads `evidence.json` and creates `aggregation.json`.

`POST /nlp/aggregate-llm`

Aggregates LLM evidence using label counts, provide basic summary.

`POST /nlp/aggregate-classifier`

Later endpoint. Aggregates classifier evidence using confidence-weighted voting. Might skip entirely tbh

### Final Report

Reads `sources.json`, `evidence.json`, and `aggregation.json`, then creates `final_report.json`.

`POST /report/generate`

Later endpoint.

### Full Pipeline

Endpoints for running the whole pipeline in one call.


`POST /pipeline/run-synthetic-llm`

Runs the full LLM pipeline from a synthetic case file.

`POST /pipeline/run-curated-llm`

Runs the full LLM pipeline from curated URLs.

`POST /pipeline/run-search-llm`

Runs the full LLM pipeline from a passenger name.

Classifier full-pipeline endpoints are not implemented yet. Might skip entirely tbh

### Debugging

`GET /debug/runs`

Lists existing run folders under `data/runs/` and shows which artifacts exist for each run.