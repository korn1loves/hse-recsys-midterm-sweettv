# HSE RecSys midterm: sweet.tv TV program recommender

**Task.** Kaggle "sweet.tv — TV program recommender": predict 5 tv_show_ids (ordered) for each of the 1,260 users in `submission.csv`, for weeks 31-42. The metric is MAP@5. Inputs are a view log for weeks 11-30 and channel schedules for weeks 11-42.

**Results** (MAP@5, validation main = 11-20→21-30 / secondary = 11-24→25-30): Hybrid 0.302 / 0.262, personal history 0.297 / 0.262, popularity 0.177 / 0.153, AR 0.175 / 0.151, MF 0.091 / 0.074, CBF 0.004 / 0.004. All numbers and protocol details are in `results/report_data.md`.

## Reproduce

1. Download the competition data into `data/sweettv-tv-program-recommender/` (`dataset11-30.csv`, `export_arh_{11-20,21-30,31-42}-final.csv`, `submission.csv`). It is not in git.
2. Create the environment:
   ```
   python3 -m venv .venv
   .venv/bin/pip install pandas numpy scipy scikit-learn matplotlib pyarrow tabulate pytest jupyter
   ```
   Exact versions are in `results/requirements-freeze.txt`.
3. Run from `src/`, in order:
   ```
   cd src
   ../.venv/bin/python audit.py           # data audit           -> results/audit.md
   ../.venv/bin/python join_report.py     # join + threshold     -> results/join_report.md, fig_overlap.png
   ../.venv/bin/python baselines.py       # Phase A references   -> results/baselines.md/.csv
   ../.venv/bin/python make_submission.py # baseline submission  -> results/submission_baseline.csv
   ../.venv/bin/python hybrid.py          # MF/CBF/AR/hybrid     -> results/phaseBC.md, comparison.csv,
                                          #   mf_grid.csv, hybrid_grid.csv, examples.md, submission_final.csv
   cd .. && .venv/bin/python -m pytest -q tests
   ```
   The parquet cache (`results/cache/`) is rebuilt automatically and is gitignored. The full pipeline takes about 3-4 minutes on an Apple M4.

## File map

| path | content |
|---|---|
| `src/data.py` | CSV loading with explicit timestamp formats, ISO weeks, parquet cache |
| `src/join.py` | view × schedule interval join, `watched()` threshold, aggregation |
| `src/metrics.py` | Kaggle-style AP@K / MAP@K |
| `src/baselines.py` | splits, targets, candidate pool, fallback, popularity / personal references |
| `src/scoring.py` | shared score-matrix → top-5 + fallback, bootstrap CIs, min-max |
| `src/mf.py` | implicit ALS (Hu-Koren-Volinsky) in numpy/scipy |
| `src/cbf.py` | TF-IDF content-based filtering on schedule attributes |
| `src/ar.py` | item-to-item association rules |
| `src/hybrid.py` | Phase B/C driver: grids, fusion, comparison, examples, final submission |
| `src/audit.py`, `src/join_report.py`, `src/make_submission.py` | Phase A reports and baseline submission |
| `tests/test_metrics.py` | MAP@5 unit tests |
| `results/report_data.md` | all numbers for the course report |
| `results/phaseA_report.md`, `results/phaseBC.md`, `results/examples.md` | phase reports, example users |
| `results/submission_final.csv`, `results/submission_baseline.csv` | Kaggle-format submissions (not submitted by the scripts) |

## Environment

Apple M4, 16 GB RAM, macOS 26.6.2, Python 3.13.5, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1 (`results/environment.md`).

`.venv`, `.gitignore` and `src/data.py` were created in an earlier, abandoned Claude Code session on 2026-10-21. They were reviewed and reused; all results were regenerated on 2026-10-24.
