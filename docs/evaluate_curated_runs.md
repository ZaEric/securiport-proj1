# Evaluate Curated Runs

This document explains how to use:

```text
scripts/evaluate_curated_runs.py
```

The script compares curated expected sentiment labels against the aggregation baseline sentiment from completed runs.

## Purpose

Use this script after running the pipeline through:

```text
sources.json -> chunks.json -> evidence.json -> aggregation.json
```

It helps answer:

- which curated cases have completed aggregation outputs?
- does `aggregation.json` match the expected sentiment in `curated_urls.json`?
- how many sources and evidence snippets were used?
- which cases are mixed, insufficient, or mismatched?

Important: this script evaluates the aggregation baseline. It does not evaluate a final report result. The final report step is still separate.

## Required Inputs

The script reads expected labels from:

```text
data/curated/curated_urls.json
```

It reads run artifacts from:

```text
data/runs/<run_id>/input.json
data/runs/<run_id>/sources.json
data/runs/<run_id>/evidence.json
data/runs/<run_id>/aggregation.json
```

Only runs with `input_mode` set to `curated_urls` are evaluated.

If there are multiple runs for the same person, the script uses the latest run with an `aggregation.json`.

## Basic Usage

From the project root:

```powershell
python scripts/evaluate_curated_runs.py
```

This prints a table:

```text
Target             Expected  Baseline  Match  Evidence  Sources  Mixed  Insufficient
-----------------  --------  --------  -----  --------  -------  -----  ------------
Simone Biles       positive  positive  yes    12        3        no     no
Malala Yousafzai   positive  positive  yes    5         1        no     no
Elizabeth Holmes   negative  negative  yes    9         2        yes    no
Sam Bankman-Fried  negative  negative  yes    4         1        no     no
Albert Einstein    neutral   positive  no     2         1        no     no
```

## Write JSON Summary

To write a machine-readable summary:

```powershell
python scripts/evaluate_curated_runs.py --write-json
```

Default output path:

```text
data/evaluation/curated_runs_summary.json
```

Use a custom output path:

```powershell
python scripts/evaluate_curated_runs.py --write-json --output-path "data/evaluation/my_summary.json"
```

## Output Fields

Each result includes:

- `target_name`: person being evaluated
- `run_id`: selected run ID
- `expected_sentiment`: label from `curated_urls.json`
- `baseline_sentiment`: `aggregation.json -> overall_result -> baseline_sentiment`
- `match`: whether expected and baseline sentiment match
- `source_count`: number of sources in `sources.json`
- `evidence_count`: number of evidence snippets in `evidence.json`
- `mixed_evidence`: `aggregation.json -> overall_result -> mixed_evidence`
- `insufficient_evidence`: `aggregation.json -> overall_result -> insufficient_evidence`
- `sentiment_counts`: aggregation-level positive, neutral, and negative counts
- `notes`: short explanation for missing or mismatched results

## Current Interpretation

With the current curated runs, the baseline aggregation result is:

```text
5 evaluable cases
4 matched expected sentiment
1 mismatched expected sentiment
```

The known mismatch is:

```text
Albert Einstein: expected=neutral, baseline=positive
```

This is expected under the current aggregation rule because the Nobel Prize profile contains positive achievement evidence. Aggregation is count-based and treats positive evidence as the baseline when there is positive evidence and no negative evidence.

## Recommended Workflow

1. Run or update curated pipeline artifacts.
2. Run:

```powershell
python scripts/evaluate_curated_runs.py
```

3. Review mismatches and low-evidence cases.
4. If needed, update curated URLs or expected labels.
5. Write a summary:

```powershell
python scripts/evaluate_curated_runs.py --write-json
```

6. Use the summary in project documentation or presentation notes.
