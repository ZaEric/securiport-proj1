# securiport-proj1

## Setup

```bash
conda env create -f environment.yml
conda activate securiport-proj1
```

## Directory Setup

```
data/
├── curated/
│   └── curated_urls.json
├── synthetic/
│   ├── index.json
│   └── passengers/
│       ├── synthetic_negative_001.json
│       ├── synthetic_positive_001.json
│       └── synthetic_neutral_001.json
└── runs/
    └── 2026-06-27_001_jane_doe/
        ├── input.json
        ├── sources.json
        ├── chunks.json
        ├── evidence.json
        ├── aggregation.json
        └── final_report.json
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