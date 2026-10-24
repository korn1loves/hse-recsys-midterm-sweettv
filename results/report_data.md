# Report data: sweet.tv TV program recommender (HSE RecSys midterm)

Every number here was produced by the scripts in `src/` on 2026-10-24 (see the generated files: `audit.md`, `join_report.md`, `baselines.md`, `phaseBC.md`, `examples.md`, `*.csv`).

**Provenance note.** `.venv`, `.gitignore` and `src/data.py` were created in an earlier, abandoned Claude Code session on 2026-10-21. In this session they were reviewed and reused. The parquet cache from that session was deleted and rebuilt from the raw CSVs, and all results were regenerated.

## 1. Environment, seeds, runtime

- Apple M4 (10 cores, 16 GB RAM), macOS 26.6.2, Python 3.13.5 (`.venv`).
- Packages: pandas 3.0.6, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, pyarrow 26.0.0. Full list in `results/requirements-freeze.txt`; details in `results/environment.md`.
- Seeds:
  - ALS initialisation: `np.random.default_rng(0)`.
  - Bootstrap: `default_rng(0)`, 1000 resamples.
  - Hand-checked join example: `default_rng(0)`.
  - Everything else is deterministic; ties are broken by smaller tv_show_id.
- End-to-end runtime:
  - parquet cache build: ~12 s
  - join: 4.3 s
  - Phase A baselines: 13.5 s
  - Phase B/C driver: 154 s, including the 8-config ALS grid and the 120-config hybrid grid.

## 2. Data and join

- **View log:** 5,821,987 rows (17 exact duplicates dropped), 4,914 users, 202 channels, 2020-03-09 .. 2020-07-26, ISO weeks 11-30. Minimum `duraton` is 61 s; the median view is 396 s.
- **Schedules:** 1,074,235 / 1,074,394 / 1,295,378 rows (11-20 / 21-30 / 31-42). Only **10.76% / 11.31% / 11.18%** of rows have tv_show_id ≠ 0, which is about 16-17% of airtime.
- **Formats:** the log is `YYYY-MM-DD HH:MM:SS` and the schedules are `DD.MM.YYYY HH:MM:SS`. Both were parsed with explicit formats, and ISO weeks were verified (week 11 starts on Monday 2020-03-09, week 31 on Monday 2020-07-27).
- **Join rule:** same channel (vsetv_id = channel_id) and overlap of [view start, view stop] with [airing start, airing start + duration]. Overlap seconds = min(ends) − max(starts) > 0, summed per (user, airing, ISO week of the view start) and capped at the airing duration.
- **Where viewing time falls:**
  - labelled airings: **52.36%**
  - unlabelled airtime: **40.68%**
  - no airing: **6.96%**
- **Handling of id 0:** unlabelled airtime is not an item. It is excluded from history, targets, the pool and every model.
- **Watched threshold:** overlap ≥ **300 s** per (user, airing). Overlaps under 300 s are 31% of labelled user×airing pairs but only 2.9% of the seconds; they are zapping.
  - The threshold defines the target.
  - Model history uses **0 s** (all overlap). On validation, 0 s ≈ 300 s > 900 s: personal history scored 0.29708 / 0.29497 / 0.28301 on main.

## 3. Protocol (identical for every method)

- **Splits:**
  - main: history weeks 11-20 → target 21-30, 1,246 eval users
  - secondary: history 11-24 → target 25-30, 1,244 eval users
  - Eval users are submission users with a non-empty target. All 1,260 submission users have log rows.
- **Target (hypothesis):** per user, the top 5 labelled shows by watched seconds (airings with ≥ 300 s overlap) in the target window.
- **Candidate pool:** labelled shows with an airing starting in a target week, taken from the schedule only. Sizes: 6,376 (main), 5,013 (secondary), 6,632 (final, weeks 31-42). No target item falls outside the pool.
- **Repeats are allowed.** 66.4% (main) and 62.3% (secondary) of eval users' target items were already watched by the user in history.
- **Cold start / filling:** each method ranks its *valid* pool items (score > 0; for MF, all pool items for users with history) by score, breaking ties by smaller id. It then fills up to 5 with the fallback: pool popularity by watched seconds over all history users, skipping duplicates. This guarantees exactly 5 distinct non-zero pool ids, which is asserted for every method and split.
- **Metric:** MAP@5 with Kaggle semantics (AP@5 = Σ P(k)·rel(k) / min(|truth|, 5)), unit-tested in `tests/test_metrics.py`.
- **Training data:** only history-week viewing is used to fit. The target-window schedule supplies only the pool and item attributes.

## 4. Methods and hyperparameters

- **Popularity (reference):** pool shows ranked by total watched seconds over all users in history.
  - The recency variant weights week w by 0.5^((last_week − w)/1).
  - Phase A also tested half-lives of 2 and 4 weeks, unrestricted popularity and popularity by number of users (`baselines.md`).
- **Personal history (reference):** the user's pool shows ranked by total history seconds, filled with the fallback. Recency weighting hurt, so none is used.
- **MF, implicit ALS** (Hu-Koren-Volinsky, `src/mf.py`, numpy/scipy):
  - Matrix: all 4,841 users with history × shows (history ∪ pool).
  - Preference p = 1 on observed pairs; confidence c = 1 + α·log1p(seconds/60).
  - Exact per-row normal-equation solves, 15 iterations, seed 0.
  - Grid on main, results below. **Chosen: 32 factors, reg 1.0, α 1.0** (main 0.09094; secondary confirmation 0.07367).
- **CBF** (`src/cbf.py`):
  - Item tokens: `cat:` category, `g:` genres 1-3, `dir:` director names, `act:` actor names (split on commas), `yr:` decade of year_of_production.
  - TF-IDF with min_df = 2 and L2 normalisation: 8,234 items × 5,362 features (main).
  - Item attributes come from the first non-null value per show in the schedules.
  - User profile = log1p(seconds)-weighted mean of the user's history item vectors, then L2-normalised. Score = cosine against pool items.
- **Association rules** (`src/ar.py`):
  - Transactions: per user, the shows with ≥ 300 s total history seconds (4,834 transactions).
  - Rule a→b is kept if support ≥ 0.01 and confidence ≥ 0.1, giving 344,273 rules on main.
  - Score(u, b) = Σ conf(a→b) over a in the user's set.
- **Hybrid** (`src/hybrid.py`):
  - Per-user min-max of each component's scores over the pool, then a weighted sum.
  - The valid set is the union of the valid sets of components with weight > 0; the result is filled with the fallback.
  - Grid on main: w_pers = 1, w_mf ∈ {0, .05, .1, .25, .5, 1}, w_cbf ∈ {0, .05, .1, .25}, w_ar ∈ {0, .05, .1, .25, .5} (120 configs).
  - **Chosen: w_pers 1, w_mf 1, w_cbf 0, w_ar 0.25.**

MF grid (main, MAP@5):

| factors | reg | alpha | MAP@5 | fit s |
|---|---|---|---|---|
| 32 | 1.0 | 1 | **0.09094** | 14.6 |
| 32 | 0.1 | 1 | 0.09089 | 3.0 |
| 64 | 1.0 | 1 | 0.06659 | 6.5 |
| 64 | 0.1 | 1 | 0.06658 | 38.3 |
| 32 | 1.0 | 10 | 0.01986 | 15.6 |
| 32 | 0.1 | 10 | 0.01971 | 4.0 |
| 64 | 1.0 | 10 | 0.01185 | 5.5 |
| 64 | 0.1 | 10 | 0.01158 | 16.0 |

Fit times vary from 3 to 38 s for the same workload. That reflects machine load during the run, not the model size.

Hybrid grid, top 5 (main): (mf 1, cbf 0, ar .25) 0.30178; (.25, 0, .25) 0.30175; (.1, 0, .25) 0.30144; (.05, 0, .25) 0.30141; (.5, 0, .25) 0.30141. Personal + AR alone (mf 0, cbf 0, ar .25) scores 0.30089. Full grid: `results/hybrid_grid.csv`.

## 5. Comparison (MAP@5; 95% bootstrap CI over users, 1000 resamples, seed 0)

| method | main | main 95% CI | secondary | secondary 95% CI | fit s (main) | predict s (main) |
|---|---|---|---|---|---|---|
| Hybrid (pers 1 + MF 1 + AR 0.25) | **0.30178** | [0.2874, 0.3148] | 0.26164 | [0.2493, 0.2753] | 4.10 | 0.19 |
| Personal history | 0.29708 | [0.2830, 0.3102] | **0.26248** | [0.2497, 0.2769] | 0.08 | 0.04 |
| Popularity (pool, half-life 1 wk) | 0.18256 | [0.1722, 0.1926] | 0.15799 | [0.1485, 0.1679] | 0.04 | – |
| Popularity (pool, seconds) | 0.17685 | [0.1666, 0.1865] | 0.15295 | [0.1440, 0.1625] | 0.05 | – |
| Association rules | 0.17543 | [0.1653, 0.1847] | 0.15143 | [0.1425, 0.1608] | 0.29 | 0.10 |
| MF (ALS) | 0.09094 | [0.0830, 0.0993] | 0.07367 | [0.0669, 0.0803] | 2.98 | 0.08 |
| CBF | 0.00383 | [0.0025, 0.0052] | 0.00417 | [0.0028, 0.0056] | 0.75 | 0.20 |

- **Paired bootstrap, Hybrid − Personal on main:** +0.00470, 95% CI [+0.00015, +0.00887]. On secondary the hybrid is 0.00084 *below* personal history.
- **Interpretation:** the hybrid gain is small, barely significant on main and not confirmed on secondary. Repeat viewing dominates this target: 62-66% of target items were already watched.
- **Why MF and CBF fail alone:**
  - MF models *whether* a user watches a show, not how many seconds. Its top items are unwatched films (example below), while the target ranks by seconds.
  - CBF pushes films that share actors and directors, because those token fields are the richest. Few films reach a user's top 5 by seconds.

## 6. Final submission

- **File:** `results/submission_final.csv`.
- **Strategy:** the Hybrid, selected on main as pre-registered (pers 1, MF 1 with 32 factors / reg 1 / α 1, AR 0.25).
- **Training:** retrained on weeks 11-30 (history threshold 0 s). The pool is the 6,632 labelled shows starting in weeks 31-42. CBF attributes come from all three schedules, although CBF has weight 0.
- **Validation:** same header, 1,260 rows in template order, LF line endings, no trailing newline, 5 distinct non-zero pool ids per row. All checked by `make_submission.validate()`.
- **Overlap with the baseline:** 16.9% of rows are identical to `submission_baseline.csv` (personal history), and on average 81.8% of the items in a row are shared.
- Because the gain is not confirmed on secondary, the baseline file is an equally defensible submission.

## 7. Limitations (with numbers)

- **Unlabelled airtime:** only 10.8-11.3% of schedule rows are labelled. 40.7% of viewing time falls on unlabelled airtime and is ignored.
- **New shows:** 1,437 of the 6,636 labelled shows airing in weeks 31-42 never appear in the 11-30 schedules. Personal history, MF and AR cannot recommend them; only CBF could, and it is weak.
- **Target definition:** the competition's preference rule is unpublished. Our target is a hypothesis (top 5 by seconds, ≥ 300 s). Method ranking was stable for target thresholds of 0, 300 and 900 s (Phase A).
- **Activity drop:** activity falls in weeks 20-27 (as low as 2,406 active users in week 26, vs about 4,600 in weeks 11-19).
- **Horizon mismatch:** validation horizons are 10 and 6 weeks; the test horizon is 12.
- **Time alignment:** the time-zone check is weak. View starts align best with airing starts at shift 0, but only 1.81% vs 1.73-1.78% for nearby shifts.
- **Coverage gaps:** 6.96% of viewing time hits no airing, and 10 log channels are missing from at least one schedule.
- **Outlier users:** the heavy example user logs 1,539 h in 10 weeks (about 22 h/day), probably a TV left running. Such users weigh heavily in seconds-based popularity.
- **Limited tuning:** MF was tuned only on the prescribed 8-point grid. Larger regularisation or a seconds-aware objective might help. CBF used a single configuration.

## 8. Example users (main split; same users for every method)


Selection rule: among main-split eval users with any history in weeks 11-20, heavy = maximum total labelled overlap seconds in history, light = minimum.

### Heavy viewer: user 13571199046623737884

History weeks 11-20: 5,541,593 s (1539.3 h) over 837 distinct labelled shows.

Top history shows:

|   tv_show_id | title                                     | category   |   seconds |
|-------------:|:------------------------------------------|:-----------|----------:|
|      2400480 | 'ТСН. Неделя'.                            | Инфо       |    154208 |
|     10003159 | Т/с 'Ничто не случается дважды', 1 и 2 с. | Сериалы    |    119238 |
|      2400508 | Сегодня. Итоги с Олегом Панютой.          | Инфо       |    116342 |
|       240081 | Факты. День.                              | Инфо       |    115169 |
|     12001682 | М/с 'Губка Боб Квадратные Штаны'.         | Детям      |    107941 |

Truth (top-5 watched in weeks 21-30):

|   tv_show_id | title                             | category       |   seconds |
|-------------:|:----------------------------------|:---------------|----------:|
|       240081 | Факты. День.                      | Инфо           |     17563 |
|       200352 | 'Вечерний квартал'.               | Развлечения    |     10817 |
|      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |      9756 |
|      2500413 | Гражданская оборона.              | Инфо           |      9678 |
|      2400480 | 'ТСН. Неделя'.                    | Инфо           |      9600 |

**Popularity (pool, seconds)** (AP@5 = 0.520):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |       240081 | Факты. День.                                              | Инфо           | ✓     |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      4 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      5 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |

**Popularity (pool, half-life 1 wk)** (AP@5 = 0.600):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |       240081 | Факты. День.                                              | Инфо           | ✓     |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |

**Personal history** (AP@5 = 0.300):

|   rank |   tv_show_id | title                                     | category   | hit   |
|-------:|-------------:|:------------------------------------------|:-----------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                            | Инфо       | ✓     |
|      2 |     10003159 | Т/с 'Ничто не случается дважды', 1 и 2 с. | Сериалы    |       |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.          | Инфо       |       |
|      4 |       240081 | Факты. День.                              | Инфо       | ✓     |
|      5 |     12001682 | М/с 'Губка Боб Квадратные Штаны'.         | Детям      |       |

**MF (ALS)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                               | category   | hit   |
|-------:|-------------:|:------------------------------------|:-----------|:------|
|      1 |     90078209 | Х/ф 'Хоббит: Битва пяти воинств'.   | Фильмы     |       |
|      2 |     90074731 | Х/ф 'Мстители'.                     | Фильмы     |       |
|      3 |     90069354 | Х/ф 'Операция 'Валькирия'.          | Фильмы     |       |
|      4 |     90073874 | Х/ф 'Ковбои против пришельцев'.     | Фильмы     |       |
|      5 |    120060773 | М/ф 'Алеша Попович и Тугарин Змей'. | Детям      |       |

**CBF** (AP@5 = 0.000):

|   rank |   tv_show_id | title                                     | category   | hit   |
|-------:|-------------:|:------------------------------------------|:-----------|:------|
|      1 |     90082800 | Х/ф 'Волшебная малина'                    | Фильмы     |       |
|      2 |     90080935 | Х/ф 'Уличный боец'.                       | Фильмы     |       |
|      3 |     90079706 | Х/ф 'Зип и Зап. Клуб стеклянных шариков'. | Фильмы     |       |
|      4 |     90082298 | Х/ф 'Продюсер'.                           | Фильмы     |       |
|      5 |     90081599 | Х/ф 'Не игра'.                            | Фильмы     |       |

**Assoc. rules** (AP@5 = 0.550):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           | ✓     |
|      2 |       240081 | Факты. День.                      | Инфо           | ✓     |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |
|      4 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное | ✓     |
|      5 |      2400467 | 'Викна-Новости'.                  | Инфо           |       |

**Hybrid** (AP@5 = 0.300):

|   rank |   tv_show_id | title                                     | category   | hit   |
|-------:|-------------:|:------------------------------------------|:-----------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                            | Инфо       | ✓     |
|      2 |     10003159 | Т/с 'Ничто не случается дважды', 1 и 2 с. | Сериалы    |       |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.          | Инфо       |       |
|      4 |       240081 | Факты. День.                              | Инфо       | ✓     |
|      5 |     12001682 | М/с 'Губка Боб Квадратные Штаны'.         | Детям      |       |

**Final strategy (Hybrid), weeks 31-42 Top-5:**

|   rank |   tv_show_id | title                                     | category   |
|-------:|-------------:|:------------------------------------------|:-----------|
|      1 |      2400480 | 'ТСН. Неделя'.                            | Инфо       |
|      2 |     10003159 | Т/с 'Ничто не случается дважды', 1 и 2 с. | Сериалы    |
|      3 |       240081 | Факты. День.                              | Инфо       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.          | Инфо       |
|      5 |     12001682 | М/с 'Губка Боб Квадратные Штаны'.         | Детям      |

### Light viewer: user 970670590731358344

History weeks 11-20: 1,983 s (0.6 h) over 3 distinct labelled shows.

Top history shows:

|   tv_show_id | title                  | category   |   seconds |
|-------------:|:-----------------------|:-----------|----------:|
|      2400480 | 'ТСН. Неделя'.         | Инфо       |      1008 |
|       240081 | Факты. День.           | Инфо       |       697 |
|      6700482 | 'Секретные материалы'. | Другое     |       278 |

Truth (top-5 watched in weeks 21-30):

|   tv_show_id | title                                                        | category    |   seconds |
|-------------:|:-------------------------------------------------------------|:------------|----------:|
|      5800509 | Реальная мистика.                                            | Развлечения |      5859 |
|     12001863 | М/с 'Смешарики'.                                             | Детям       |      4200 |
|     12002733 | М/с 'Буба'.                                                  | Детям       |      3094 |
|     12002771 | М/с 'Черепашки-ниндзя', 3 сезон, 12 с. 'Битва за Нью-Йорк -  | Детям       |      2700 |
|     12001734 | М/с 'Волшебные покровители', 154 с. 'Хочу это вырезать / Мэр | Детям       |      2400 |

**Popularity (pool, seconds)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           |       |
|      2 |       240081 | Факты. День.                                              | Инфо           |       |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      4 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      5 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное |       |

**Popularity (pool, half-life 1 wk)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           |       |
|      2 |       240081 | Факты. День.                                              | Инфо           |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |

**Personal history** (AP@5 = 0.000):

|   rank |   tv_show_id | title                                                     | category    | hit   |
|-------:|-------------:|:----------------------------------------------------------|:------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо        |       |
|      2 |       240081 | Факты. День.                                              | Инфо        |       |
|      3 |      6700482 | 'Секретные материалы'.                                    | Другое      |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо        |       |
|      5 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |       |

**MF (ALS)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           |       |
|      2 |       240081 | Факты. День.                      | Инфо           |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |
|      5 |      2400467 | 'Викна-Новости'.                  | Инфо           |       |

**CBF** (AP@5 = 0.000):

|   rank |   tv_show_id | title                 | category   | hit   |
|-------:|-------------:|:----------------------|:-----------|:------|
|      1 |       240081 | Факты. День.          | Инфо       |       |
|      2 |      2500261 | 'Подробности недели'. | Инфо       |       |
|      3 |      2400480 | 'ТСН. Неделя'.        | Инфо       |       |
|      4 |      5900347 | 'Деньги'.             | Инфо       |       |
|      5 |      5900319 | Абзац!                | Инфо       |       |

**Assoc. rules** (AP@5 = 0.000):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |
|      2 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |       |
|      3 |      2400467 | 'Викна-Новости'.                  | Инфо           |       |
|      4 |     10002514 | Т/с 'Кухня', 1 сезон, 1-4 с.      | Сериалы        |       |
|      5 |       200352 | 'Вечерний квартал'.               | Развлечения    |       |

**Hybrid** (AP@5 = 0.000):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           |       |
|      2 |       240081 | Факты. День.                      | Инфо           |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |
|      5 |      2400467 | 'Викна-Новости'.                  | Инфо           |       |

**Final strategy (Hybrid), weeks 31-42 Top-5:**

|   rank |   tv_show_id | title                                                        | category    |
|-------:|-------------:|:-------------------------------------------------------------|:------------|
|      1 |      5800509 | Реальная мистика.                                            | Развлечения |
|      2 |     12001863 | М/с 'Смешарики'.                                             | Детям       |
|      3 |     12002733 | М/с 'Буба'.                                                  | Детям       |
|      4 |     12001734 | М/с 'Волшебные покровители', 154 с. 'Хочу это вырезать / Мэр | Детям       |
|      5 |     12002771 | М/с 'Черепашки-ниндзя', 3 сезон, 12 с. 'Битва за Нью-Йорк -  | Детям       |

