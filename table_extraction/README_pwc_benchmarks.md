# PwC benchmark notebooks (replication guide)

This document describes how to reproduce evaluation results for the four benchmark notebooks in `table_extraction/`. Run all notebooks from the repository root (the directory that contains `data/pwc_final.json`).

## Paths and folder names

Folder names and paths in this README are the defaults used in the notebooks. Your layout may differ. Set paths in each notebook configuration cell; do not assume fixed directory names on disk.

Typical variables:

`REPO` or equivalent: repository root (where `data/pwc_final.json` lives).

`PDF_FILES_DIR`: directory of evaluation PDFs (default in notebooks: `data/pdf_files_3`). You may use `data/pdf_files` or any other folder; point `PDF_FILES_DIR` to it.

`XML_FILES_DIR` / TEI path: GROBID XML for `text_only` (default: `data/xml_files_3`). Must match how your project stores TEI files.

`TABLE_EXTRACTION_DIR`: where combination and eval Excel files are written (default: `table_extraction/` under the repo). You may redirect outputs or copy results into subfolders such as `gliner_excels/` or `ollama_excels/`; the notebooks do not require those subfolder names.

`CORPUS_TAG`: string suffix embedded in some Excel filenames (default: `pdf_files`). It does not have to match the PDF folder name; it only labels exports.

`LLM_CACHE_DIR`: Ollama response cache (dataset notebooks: `llm_ollama_dataset_cache_{CORPUS_TAG}`; metric notebooks: `llm_ollama_cache_{CORPUS_TAG}`). The cache folder name follows `CORPUS_TAG`, not necessarily the PDF directory name.

LightOnOCR table JSON files must live alongside PDFs (or wherever the notebook resolves them): `<pdf_stem>_lightonocr.json` per paper. The stem must match the PDF filename.

Ground truth file: `data/pwc_final.json` by default (`PWC_GT_JSON` in config). You may use another basename if the notebook variable is updated consistently.

## Overview

| Notebook | Tool | PwC field | Combination Excel (pattern) | Eval / audit Excel (default names) |
|----------|------|-----------|------------------------------|-------------------------------------|
| `table_dataset_extraction_and_pwc_benchmark_gliner.ipynb` | GLiNER2 + LightOnOCR | `Datasets` | `gliner2_lightonocr_dataset_combinations_{mode}_{CORPUS_TAG}.xlsx` | `gliner_dataset_eval_allpapers.xlsx`, `gliner_dataset_audit_allpapers.xlsx` |
| `table_dataset_extraction_and_pwc_benchmark_ollama.ipynb` | Ollama | `Datasets` | `ollama_{model}_lightonocr_dataset_combinations_{CORPUS_TAG}.xlsx` | `ollama_dataset_eval_allpapers.xlsx`, `ollama_dataset_audit_allpapers.xlsx` |
| `table_metric_extraction_and_pwc_benchmark_gliner.ipynb` | GLiNER2 + LightOnOCR | `Metrics` | `gliner2_lightonocr_combinations_{mode}_{CORPUS_TAG}.xlsx` | `gliner_metric_eval_allpapers.xlsx`, `gliner_metric_audit_allpapers.xlsx` |
| `table_metric_extraction_and_pwc_benchmark_ollama.ipynb` | Ollama | `Metrics` | `ollama_{model}_lightonocr_combinations_{CORPUS_TAG}.xlsx` | `ollama_metric_eval_allpapers.xlsx`, `ollama_metric_audit_allpapers.xlsx` |

`{model}` is the slug from `OLLAMA_MODELS` in the Ollama config (for example `qwen3_1.7b`, `llama3_8b`). `{mode}` in GLiNER combination names comes from the export mode variable in the GLiNER notebook (for example `filtered`).

## Prerequisites

Step 1: Python environment with dependencies from each notebook first cell (`AUTO_INSTALL = True`), or install manually: `pandas`, `openpyxl`, `beautifulsoup4`, GLiNER stack, `ollama` client for Ollama notebooks.

Step 2: Data layout (paths configurable as above):

`pwc_final.json` for ground truth.

PDF corpus folder with one PDF per paper.

Per-PDF LightOnOCR table cache `*_lightonocr.json` from your table OCR pipeline (for example `extract_table_context_lightonocr.ipynb` or equivalent). Required for table-based extraction.

TEI XML corpus for `text_only` mode, if you enable text evaluation.

Step 3: GLiNER notebooks: GPU recommended; Hugging Face access for `fastino/gliner2-base-v1`. Optional HF login cell or `HF_TOKEN`.

Step 4: Ollama notebooks: running Ollama daemon; models available per `OLLAMA_MODELS` (`ollama list` / `ollama pull`).

Step 5: For comparable dataset benchmarks between GLiNER and Ollama, use the same `normalize_dataset` in both notebooks (parentheses removed before normalization). Re-run evaluation after changing normalization.

## Evaluation settings (all four notebooks)

Keep these flags as in the config cells unless you run an ablation:

`FILTER_PREDICTIONS_TO_PWC_GT_DATASETS` or `FILTER_PREDICTIONS_TO_PWC_GT_VOCAB` = `True` (predictions restricted to PwC vocabulary per paper).

`EVAL_ONLY_PWC_MATCHED_PAPERS` = `True` (evaluate only papers that match an entry in `pwc_final.json` for the chosen field).

`MATCH_THRESHOLD` = `0.50` (default in notebooks).

Three modes: `tables_only`, `text_only`, `tables_plus_text`.

`n_ground_truth` in the eval sheet is the micro count of ground-truth (paper, entity) pairs for your corpus and JSON; it depends on corpus size and PwC content. Dataset and metric benchmarks use the same counting logic on `Datasets` vs `Metrics` respectively. GLiNER and Ollama dataset runs should report the same `n_ground_truth` when they share corpus, JSON, flags, and `normalize_dataset`.

## GLiNER dataset notebook

File: `table_dataset_extraction_and_pwc_benchmark_gliner.ipynb`

Execution order (restart kernel, then run top to bottom):

Step 1: Configuration cell (`PDF_FILES_DIR`, `CORPUS_TAG`, flags).

Step 2: Optional Hugging Face login if needed.

Step 3: Corpus selection (`CORPUS_PDF_PATHS`).

Step 4: Extraction (LightOnOCR JSON, GLiNER, combination Excel, `INPUT_EXCEL`).

Step 5: Optional XML ground-truth cell (`EVALUATE_AGAINST_XML_GT = False` unless you need it).

Step 6: PwC evaluation (`gliner_dataset_eval_allpapers.xlsx`, `gliner_dataset_audit_allpapers.xlsx`). Default `REEXTRACT_TABLES_WITH_GLINER = True` refreshes table predictions from caches.

Check: sheet `evaluation` has one row per enabled mode; precision is 1.0 when PwC filtering is enabled; `n_ground_truth` is identical across modes in the same run.

## GLiNER metric notebook

File: `table_metric_extraction_and_pwc_benchmark_gliner.ipynb`

Same order as the dataset GLiNER notebook. Sheets and filenames use Metrics / `gliner_metric_*` instead of Datasets / `gliner_dataset_*`.

## Ollama dataset notebook

File: `table_dataset_extraction_and_pwc_benchmark_ollama.ipynb`

Execution order:

Step 1: Configuration (`PDF_FILES_DIR`, `LLM_CACHE_DIR`, `OLLAMA_EVAL_ALL_XLSX`, etc.).

Step 2: Check Ollama and models.

Step 3: Load LightOnOCR caches.

Step 4: Ollama extractor and table helpers.

Step 5: PwC evaluation helpers (Section 5a; needed when filtering to PwC GT).

Step 6: `run_ollama_*` definitions (Section 5b).

Step 7: Extraction loop per model (Section 5c).

Step 8: Evaluation (Section 6b).

Cache: not required to delete when only re-running evaluation after a normalization change. Delete or rename the cache directory only when you change prompts, models, or chunk settings and want new LLM calls.

## Ollama metric notebook

File: `table_metric_extraction_and_pwc_benchmark_ollama.ipynb`

Same order as the Ollama dataset notebook. Cache and combination filenames follow the metric naming pattern (no `dataset` in the combination Excel name).

## Replication checklist

After a full run, confirm:

Step 1: No errors in the last evaluation cell.

Step 2: GLiNER eval sheet has one row per enabled mode; Ollama eval has one row per model and mode you configured.

Step 3: For dataset benchmarks on the same corpus, GLiNER and Ollama show the same `n_ground_truth` on every row.

Step 4: Combination Excel files exist at the paths your config and extraction cells wrote (check printed paths in the notebook output).

Optional: maintain a combined workbook under any path you choose; `dataset_eval_allpapers.xlsx` in `table_extraction/` is only an example aggregate file.

## Related notebooks (not part of the four benchmarks)

`extract_table_context_lightonocr.ipynb` or similar: table `*_lightonocr.json` caches.

`extract_fulltext_lightonocr_pdf_files.ipynb`: full-page OCR JSON; not used by current `text_only` when TEI is used.

`evaluation_table_extraction_gliner2_lightonocr.ipynb`: older flow; use the four `table_*_extraction_and_pwc_benchmark_*` notebooks for replication.

## Troubleshooting

If `n_ground_truth` differs between GLiNER and Ollama on datasets, compare `normalize_dataset`, `PDF_FILES_DIR`, `pwc_final.json`, and flags; re-run evaluation only.

If Ollama extraction reports zero LLM calls but finishes quickly, the run used disk cache; that is valid if combination Excel files are complete.

If GLiNER import fails, check GPU/CUDA and package versions, then HF login if needed.

If eval cannot find combination files, check `TABLE_EXTRACTION_DIR`, `CORPUS_TAG`, model slugs, and whether files were moved to a custom folder after export.
