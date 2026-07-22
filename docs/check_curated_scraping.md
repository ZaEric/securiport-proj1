# Check Curated Scraping Quality

This document explains how to use:

```text
scripts/check_curated_scraping.py
```

The script checks whether curated URLs can be fetched and whether useful text can be extracted from each page.

## Purpose

Use this script after editing:

```text
data/curated/curated_urls.json
```

It helps answer:

- can each URL be fetched?
- did text extraction produce useful article text?
- does the extracted text mention the target person?
- which URLs failed or were blocked?
- how many usable sources were found for each person?

By default, the script only prints a report. It does not write new files under `data/runs`.

## Basic Usage

From the project root:

```powershell
python scripts/check_curated_scraping.py
```

This checks every person in:

```text
data/curated/curated_urls.json
```

## Check One Person

```powershell
python scripts/check_curated_scraping.py --person "Elizabeth Holmes"
```

The `--person` value must match a `person` field in `data/curated/curated_urls.json`.

## Limit URLs Per Person

```powershell
python scripts/check_curated_scraping.py --max-urls 1
```

This is useful for quick smoke tests.

## Use a Different Curated URL File

```powershell
python scripts/check_curated_scraping.py --curated-urls-file "data/curated/curated_urls.json"
```

## Write Run Artifacts

By default, the script does not write files. To write `input.json` and `sources.json`, use:

```powershell
python scripts/check_curated_scraping.py --write-runs
```

This creates:

```text
data/runs/<run_id>/input.json
data/runs/<run_id>/sources.json
```

You can also write files for only one person:

```powershell
python scripts/check_curated_scraping.py --person "Malala Yousafzai" --write-runs
```

## Output Fields

For each person, the script prints a summary like:

```text
Checking Elizabeth Holmes...
  expected=negative requested=2 sources=2 failed=0 blocked=0
  src_001: words=430 chars=2750 target_found=yes
  src_002: words=510 chars=3200 target_found=yes
```

Field meanings:

- `expected`: expected sentiment from `curated_urls.json`
- `requested`: number of URLs requested after filtering
- `sources`: number of usable extracted sources
- `failed`: number of URLs that failed fetching or extraction
- `blocked`: number of URLs blocked with HTTP `401` or `403`
- `words`: extracted article word count
- `chars`: extracted article character count
- `target_found`: whether the target name appears in extracted text

## Error Output

If a URL fails, the script prints an error line:

```text
error status=401 url=https://... message=site blocked access to this URL...
```

Common causes:

- the site blocks scraping
- the URL returns non-HTML content
- the page requires JavaScript
- the page does not mention the target person
- the page is unavailable or times out

For reproducible demos, replace blocked or unstable URLs with stable public pages.

## Recommended Workflow

1. Add or edit URLs in `data/curated/curated_urls.json`.
2. Run:

```powershell
python scripts/check_curated_scraping.py --person "Target Name"
```

3. Check `sources`, `failed`, `blocked`, `words`, and `target_found`.
4. Replace bad URLs if needed.
5. When quality looks good, run:

```powershell
python scripts/check_curated_scraping.py --person "Target Name" --write-runs
```

6. Continue with:

```text
POST /processing/chunk-llm
```

using the generated `run_id`.
