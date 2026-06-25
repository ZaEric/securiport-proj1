# securiport-proj1

## Setup

```bash
conda env create -f environment.yml
conda activate securiport-proj1
```

## Directory Setup

data/
- stores local data, curated/ holds manual urls, synthetic/ is for our made up data, output/ to store final JSON reports

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

