# securiport-proj1

## Setup

```bash
conda env create -f environment.yml
conda activate securiport-proj1
```

## Directory Setup

```
data/
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
  nlp/
  processing/
```

data/
- stores local data, curated/ holds manual urls, synthetic/ is for our made up data, output/ to store final JSON reports

data/runs
- store generated outputs from running the pipeline

src/collections/
- web scraping stack, handle search engine API calls, URL filtering, HTML text extraction

src/processing/
- data prep stage, filter sources by relevance, chunking logic, filter chunks by relevance

src/nlp/
- analysis stack, evidence snippet extraction logic (using sentiment classifier or LLM), aggregation, summarize evidence with LLM for final report

src/api/
- API layer

tests/
- unit tests

## JSON Schemas
check schema.json

## Model Hosting

I'm using Ollama API, just went on Ollama and made a free API key for now. Need to create a .env file with this to work:
```
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=your_key_here
OLLAMA_MODEL=whatever_model_we_choose (I'm using gpt-oss:20b)
```

Note: Ollama free tier usage limits is pretty generous. Ran evidence extraction 3 times, used like 0.6% of weekly limit (usage limits resets every week).

## Search API (for `/sources/from-search-api`)

Uses Tavily to find public URLs for a target name (Bing's public Search API was retired, so we're not using that despite what the schema example used to say). Add to `.env`:
```
TAVILY_API_KEY=your_key_here
```
Free tier: 1,000 searches/month. Get a key at https://tavily.com.

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

Later endpoint. Uses an LLM to extract exact evidence quotes and sentiment labels.

`POST /nlp/evidence-classifier`

Later endpoint. Uses a sentiment classifier to label sentence/window chunks.

### Aggregation

Reads `evidence.json` and creates `aggregation.json`.

`POST /nlp/aggregate-llm`

Later endpoint. Aggregates LLM evidence using label counts.

`POST /nlp/aggregate-classifier`

Later endpoint. Aggregates classifier evidence using confidence-weighted voting.

### Final Report

Reads `evidence.json`, and `aggregation.json`, then creates `final_report.json`.

`POST /report/generate`

Later endpoint.

### Full Pipeline

Convenience endpoints for running the whole pipeline in one call.

`POST /pipeline/run-llm`

`POST /pipeline/run-classifier`

Later endpoints.