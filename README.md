# HSE RecSys midterm: sweet.tv TV program recommender

**Task.** Kaggle "sweet.tv — TV program recommender": predict 5 tv_show_ids (ordered) for each of the 1,260 users in `submission.csv`, for weeks 31-42. The metric is MAP@5. Inputs are a view log for weeks 11-30 and channel schedules for weeks 11-42.

**Results** (MAP@5, validation main = 11-20→21-30 / secondary = 11-24→25-30): Phase D Hybrid2 0.333 / 0.282, Hybrid3 0.333 / 0.284, schedule-slot exposure 0.316 / 0.268; Phase B/C hybrid 0.302 / 0.262, personal history 0.297 / 0.262, popularity 0.177 / 0.153, AR 0.175 / 0.151, MF 0.091 / 0.074, CBF 0.004 / 0.004. All numbers and protocol details are in `results/report_data.md`.

**Selection rule.** Methods are compared on the temporal validation; among submitted files the designated final submission is the one with the highest Kaggle PRIVATE score, as the exam rules specify; this makes the reported private score optimistic (test-selected).

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
   ../.venv/bin/python phase_d.py         # exposure/rate/hybrid2 -> results/phaseD.md, phaseD_*.csv,
   ../.venv/bin/python phase_d2.py        # grid around Hybrid3   -> results/phaseD2.md, submission_hybrid{4,5}.csv
   ../.venv/bin/python phase_e.py         # daypart/channel habits, RRF -> results/phaseE.md, phaseE_manifest.csv,
                                          #   submission_e01..e07.csv
   ../.venv/bin/python phase_f.py         # fixed multi-habit configs -> results/phaseF_manifest.csv, submission_f01..f08.csv
   ../.venv/bin/python phase_g.py         # title families        -> results/phaseG.md, phaseG_manifest.csv, submission_g01..g03.csv
   ../.venv/bin/python phase_h.py         # learned re-ranker     -> results/phaseH.md, phaseH_manifest.csv, submission_h01..h03.csv
   ../.venv/bin/python phase_h2.py        # no-slot re-rankers    -> results/phaseH2.md, submission_h04..h06.csv (appended to phaseH_manifest.csv)
   ../.venv/bin/python phase_i.py         # rank-level ensembles  -> results/phaseI.md, phaseI_manifest.csv, submission_i01..i04.csv
   ../.venv/bin/python phase_j.py         # re-ranker v2          -> results/phaseJ.md, phaseJ_manifest.csv, submission_j01,j02,j04.csv
   ../.venv/bin/python phase_k.py         # bagged v2 re-ranker   -> results/phaseK.md, phaseK_manifest.csv, submission_k01..k04.csv
                                          #   submission_{exposure,hybrid2,hybrid3}.csv
   cd .. && .venv/bin/python -m pytest -q tests
   ```
   The parquet cache (`results/cache/`) is rebuilt automatically and is gitignored. The full pipeline takes about 8 minutes on an Apple M4.

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
| `src/exposure.py` | Phase D: schedule-slot exposure CBF, rate-normalised personal history |
| `src/phase_k.py` | Phase K: bagged v2 re-ranker (10 seeds, user bootstrap), blends with f03 |
| `src/phase_j.py` | Phase J: re-ranker v2 (more windows with a programmatic leak rule, top-50 candidates, extra features, graded labels) |
| `src/phase_i.py` | Phase I: rank-level ensembles (RRF-style k, Borda top-30) of hybrids and re-rankers |
| `src/rerank.py`, `src/phase_h.py`, `src/phase_h2.py` | Phase H: windows, candidate generation, features, HistGradientBoosting re-ranker, blend |
| `src/family.py`, `src/phase_g.py` | Phase G: title-family key, family personal / rate components, family fallback, ablation |
| `src/phase_f.py` | Phase F: user-specified multi-habit configs (no tuning) + manifest |
| `src/phase_e.py` | Phase E: coarse habit components, reciprocal-rank fusion, candidate submissions + manifest |
| `src/phase_d.py` | Phase D driver: new methods, hybrid re-tuning, final refits |
| `src/hybrid.py` | Phase B/C driver: grids, fusion, comparison, examples, final submission |
| `src/audit.py`, `src/join_report.py`, `src/make_submission.py` | Phase A reports and baseline submission |
| `tests/test_metrics.py` | MAP@5 unit tests |
| `results/report_data.md` | all numbers for the course report |
| `results/phaseA_report.md`, `results/phaseBC.md`, `results/examples.md` | phase reports, example users |
| `results/submission_hybrid2.csv`, `submission_hybrid3.csv`, `submission_exposure.csv` | Phase D submissions (Hybrid2 = best on main, Hybrid3 = best on secondary among near-ties) |
| `results/submission_k01..k04.csv`, `results/phaseK_manifest.csv` | Phase K candidates |
| `results/submission_j01,j02,j04.csv`, `results/phaseJ_manifest.csv` | Phase J candidates |
| `results/submission_i01..i04.csv`, `results/phaseI_manifest.csv` | Phase I ensemble candidates |
| `results/submission_h01..h06.csv`, `results/phaseH_manifest.csv` | Phase H/H2 candidate set (re-rankers with and without slot signals, blends) |
| `results/submission_g01..g03.csv`, `results/phaseG_manifest.csv` | Phase G candidate set |
| `results/submission_f01..f08.csv`, `results/phaseF_manifest.csv` | Phase F candidate set |
| `results/submission_e01..e07.csv`, `results/phaseE_manifest.csv` | Phase E candidate set |
| `results/submission_final.csv`, `results/submission_baseline.csv` | Phase B/C hybrid and Phase A personal-history submissions (none are submitted by the scripts) |

## Environment

Apple M4, 16 GB RAM, macOS 26.6.2, Python 3.13.5, pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1 (`results/environment.md`).

`.venv`, `.gitignore` and `src/data.py` were created in an earlier, abandoned Claude Code session on 2026-10-21. They were reviewed and reused; all results were regenerated on 2026-10-24.
