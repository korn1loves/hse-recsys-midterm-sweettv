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


## 9. Phase D: score improvement (src/exposure.py, src/phase_d.py)

Same protocol as before: splits, target, pool, repeat policy, fallback and MAP@5. The target-window schedule is used only for what airs on which channel and when.

**Schedule-slot CBF ("exposure").**
- A slot is a (channel, hour-of-week) pair: 168 weekly hours per channel, with Monday 00:00 = hour 0.
- **Item side:** A[slot, show] = seconds of the show's labelled airings (airings starting in a target week) overlapping each clock hour, summed over the target window.
- **User side:** P[user, slot] = seconds of *all* history viewing, labelled or not, on channel c in hour-of-week h, divided by n_history_weeks × 3600. This is the user's probability of watching that slot.
  - Views are clipped at the start of the first target week, so no target-window seconds enter.
  - With recency half-life hl, week w is weighted 0.5^((last_week − w)/hl), and the denominator becomes Σ_w weight_w × 3600.
- **Score:** P @ A (sparse), i.e. the expected seconds the user will watch the show in the target window. Valid items: score > 0. Fit takes about 2 s per variant and predict 0.04 s.

**Rate-normalised personal history.** rate(u, s) = the user's history seconds on s / s's total labelled history airtime. Score = rate × s's airtime in the target-window schedule; shows the user hasn't seen score 0. Fit takes 0.3 s.

**Hybrid re-tuning.**
- Per-user min-max of each component, then a weighted sum.
- Grid on main: w_pers ∈ {0, 1}, w_rate ∈ {0, .5, 1, 2}, w_exp ∈ {0, .25, .5, 1, 2} (using the best exposure variant, half-life 4), w_ar ∈ {0, .25}, w_mf ∈ {0, 1}. That is 159 non-zero configs.
- Configs within 0.002 of the best on main were also scored on secondary.
- **Hybrid2** (best on main): pers 1, rate 1, exposure 2, AR 0.25, MF 1.
- **Hybrid3** (within 0.002 on main, best on secondary among those): pers 1, rate 0.5, exposure 2, AR 0.25, MF 1.
- The best configs sit at w_exp = 2, the edge of the grid. Larger exposure weights were not explored because of the time budget.

**Comparison (MAP@5, 95% bootstrap CI over users, 1000 resamples, seed 0):**

| method | main | main 95% CI | secondary | secondary 95% CI |
|---|---|---|---|---|
| Hybrid2 (pers 1, rate 1, exp 2, AR .25, MF 1) | **0.33315** | [0.3180, 0.3475] | 0.28205 | [0.2697, 0.2963] |
| Hybrid3 (pers 1, rate .5, exp 2, AR .25, MF 1) | 0.33297 | [0.3178, 0.3473] | **0.28413** | [0.2715, 0.2979] |
| Exposure (half-life 4 wk) | 0.31623 | [0.3016, 0.3309] | 0.26773 | [0.2548, 0.2818] |
| Exposure (no recency) | 0.31316 | [0.2990, 0.3277] | 0.26463 | [0.2521, 0.2788] |
| Exposure (half-life 2 wk) | 0.30383 | [0.2903, 0.3183] | 0.25335 | [0.2408, 0.2670] |
| Old hybrid (pers 1, MF 1, AR .25) | 0.30178 | [0.2874, 0.3148] | 0.26164 | [0.2493, 0.2753] |
| Personal history | 0.29708 | [0.2830, 0.3102] | 0.26248 | [0.2497, 0.2769] |
| Rate-normalised personal | 0.27635 | [0.2626, 0.2895] | 0.22782 | [0.2167, 0.2402] |
| Popularity (pool, seconds) | 0.17685 | [0.1666, 0.1865] | 0.15295 | [0.1440, 0.1625] |
| Association rules | 0.17543 | [0.1653, 0.1847] | 0.15143 | [0.1425, 0.1608] |
| MF (ALS) | 0.09094 | [0.0830, 0.0993] | 0.07367 | [0.0669, 0.0803] |
| CBF (attributes; Phase B/C) | 0.00383 | [0.0025, 0.0052] | 0.00417 | [0.0028, 0.0056] |

**Paired bootstrap vs the old hybrid (mean difference, 95% CI):**

| | main | secondary |
|---|---|---|
| Hybrid2 | +0.03136 [+0.02561, +0.03742] | +0.02041 [+0.01538, +0.02538] |
| Hybrid3 | +0.03119 [+0.02540, +0.03698] | +0.02249 [+0.01737, +0.02724] |

Both improvements are significant on both splits.

**Why exposure works.** It is the first method that uses *when* a show airs in the target window together with the user's channel and hour habits. That includes 40.7% of viewing time on unlabelled airtime, which no item-feedback model can use. It also scores new shows: any show airing in a slot the user habitually watches gets a score, even if the user never watched it.

**Final refits.** History weeks 11-30 (exposure profile from weeks 11-30 views, all validated), pool and airtime from the 6,632 labelled shows starting in weeks 31-42. Refit takes 13 s; the whole Phase D run takes 72 s.
- `results/submission_exposure.csv`: exposure with half-life 4 alone.
- `results/submission_hybrid2.csv`: Hybrid2.
- `results/submission_hybrid3.csv`: Hybrid3.

Each file passes the same checks as before: template header, 1,260 rows in template order, 5 distinct non-zero pool ids, LF line endings, no trailing newline.


#### Heavy viewer: user 13571199046623737884; truth = [240081, 200352, 6500479, 2500413, 2400480]

**Exposure (half-life 4 wk)** (AP@5 = 0.333):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |      2400467 | 'Викна-Новости'.                                          | Инфо           |       |

**Exposure (no recency)** (AP@5 = 0.333):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |      2400467 | 'Викна-Новости'.                                          | Инфо           |       |

**Rate-normalised personal** (AP@5 = 0.453):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           | ✓     |
|      2 |       700389 | Кто сверху?, 6 сезон, 7 эп.       | Развлечения    |       |
|      3 |       240081 | Факты. День.                      | Инфо           | ✓     |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |
|      5 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное | ✓     |

**Hybrid2** (AP@5 = 0.520):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |
|      3 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |       240081 | Факты. День.                                              | Инфо           | ✓     |

**Hybrid3** (AP@5 = 0.453):

|   rank |   tv_show_id | title                                                     | category       | hit   |
|-------:|-------------:|:----------------------------------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           | ✓     |
|      2 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |       |
|      3 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное | ✓     |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |       |
|      5 |       240081 | Факты. День.                                              | Инфо           | ✓     |

**Hybrid2, final weeks 31-42 Top-5:**

|   rank |   tv_show_id | title                                                     | category       |
|-------:|-------------:|:----------------------------------------------------------|:---------------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо           |
|      2 |       240081 | Факты. День.                                              | Инфо           |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо           |
|      4 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения    |
|      5 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп.                         | Познавательное |

#### Light viewer: user 970670590731358344; truth = [5800509, 12001863, 12002733, 12002771, 12001734]

**Exposure (half-life 4 wk)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                       | category   | hit   |
|-------:|-------------:|:----------------------------|:-----------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.              | Инфо       |       |
|      2 |      6700482 | 'Секретные материалы'.      | Другое     |       |
|      3 |       240081 | Факты. День.                | Инфо       |       |
|      4 |     90079798 | Х/ф 'Очень плохие мамочки'. | Фильмы     |       |
|      5 |     90072303 | Х/ф 'Ешь, молись, люби'.    | Фильмы     |       |

**Exposure (no recency)** (AP@5 = 0.000):

|   rank |   tv_show_id | title                       | category   | hit   |
|-------:|-------------:|:----------------------------|:-----------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.              | Инфо       |       |
|      2 |      6700482 | 'Секретные материалы'.      | Другое     |       |
|      3 |       240081 | Факты. День.                | Инфо       |       |
|      4 |     90079798 | Х/ф 'Очень плохие мамочки'. | Фильмы     |       |
|      5 |     90072303 | Х/ф 'Ешь, молись, люби'.    | Фильмы     |       |

**Rate-normalised personal** (AP@5 = 0.000):

|   rank |   tv_show_id | title                                                     | category    | hit   |
|-------:|-------------:|:----------------------------------------------------------|:------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                                            | Инфо        |       |
|      2 |       240081 | Факты. День.                                              | Инфо        |       |
|      3 |      6700482 | 'Секретные материалы'.                                    | Другое      |       |
|      4 |      2400508 | Сегодня. Итоги с Олегом Панютой.                          | Инфо        |       |
|      5 |        20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения |       |

**Hybrid2** (AP@5 = 0.000):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           |       |
|      2 |      6700482 | 'Секретные материалы'.            | Другое         |       |
|      3 |       240081 | Факты. День.                      | Инфо           |       |
|      4 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |       |
|      5 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |

**Hybrid3** (AP@5 = 0.000):

|   rank |   tv_show_id | title                             | category       | hit   |
|-------:|-------------:|:----------------------------------|:---------------|:------|
|      1 |      2400480 | 'ТСН. Неделя'.                    | Инфо           |       |
|      2 |      6700482 | 'Секретные материалы'.            | Другое         |       |
|      3 |       240081 | Факты. День.                      | Инфо           |       |
|      4 |      6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное |       |
|      5 |      2400508 | Сегодня. Итоги с Олегом Панютой.  | Инфо           |       |

**Hybrid2, final weeks 31-42 Top-5:**

|   rank |   tv_show_id | title                            | category    |
|-------:|-------------:|:---------------------------------|:------------|
|      1 |      5800509 | Реальная мистика.                | Развлечения |
|      2 |     12001863 | М/с 'Смешарики'.                 | Детям       |
|      3 |      2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо        |
|      4 |       200506 | Звездный путь.                   | Развлечения |
|      5 |     12002733 | М/с 'Буба'.                      | Детям       |


## 10. Phase D2: robustness grid around Hybrid3

**Context, reported by the user from their manual Kaggle submissions (private MAP@5, not computed in this repo):** Hybrid3 0.26923, Hybrid2 0.26807, Hybrid1 0.26507, personal history 0.24821, exposure alone 0.24023. Validation over-estimated exposure on the test weeks. These numbers were used only as a reason to be conservative, never for tuning.


Grid: {'w_exp': [0.5, 1, 2, 3, 4], 'hl': [None, 4, 8], 'w_rate': [0.25, 0.5, 1], 'w_pers': [1, 2]} with AR 0.25 and MF 1 fixed; 90 configs. Selection rule: rank by the mean of main and secondary MAP@5 (secondary is closer to the test horizon). Kaggle scores were NOT used for tuning.

Hybrid3 reference: main 0.33297, secondary 0.28413, mean 0.30855.

Top 10 (Δ = paired bootstrap mean difference vs Hybrid3, 95% CI, 1000 resamples, seed 0):

|   w_pers |   w_rate |   w_exp |      hl |    w_ar |    w_mf |   MAP@5_main |   MAP@5_secondary |    mean | Δ vs H3 main                  | Δ vs H3 secondary             |
|---------:|---------:|--------:|--------:|--------:|--------:|-------------:|------------------:|--------:|:------------------------------|:------------------------------|
|        2 |  1.00000 | 3.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33361 |           0.28435 | 0.30898 | +0.00064 [-0.00153, +0.00263] | +0.00021 [-0.00150, +0.00186] |
|        1 |  1.00000 | 3.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33321 |           0.28411 | 0.30866 | +0.00024 [-0.00151, +0.00199] | -0.00002 [-0.00142, +0.00148] |
|        2 |  0.50000 | 4.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33307 |           0.28418 | 0.30863 | +0.00010 [-0.00165, +0.00180] | +0.00005 [-0.00158, +0.00173] |
|        1 |  0.50000 | 2.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33297 |           0.28413 | 0.30855 | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
|        2 |  1.00000 | 4.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33291 |           0.28387 | 0.30839 | -0.00006 [-0.00166, +0.00150] | -0.00026 [-0.00172, +0.00130] |
|        2 |  0.25000 | 4.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33241 |           0.28430 | 0.30836 | -0.00056 [-0.00238, +0.00131] | +0.00017 [-0.00164, +0.00201] |
|        1 |  1.00000 | 4.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33210 |           0.28416 | 0.30813 | -0.00087 [-0.00303, +0.00124] | +0.00003 [-0.00218, +0.00222] |
|        1 |  0.50000 | 3.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33263 |           0.28338 | 0.30801 | -0.00034 [-0.00234, +0.00158] | -0.00075 [-0.00253, +0.00112] |
|        2 |  0.50000 | 3.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33230 |           0.28348 | 0.30789 | -0.00067 [-0.00214, +0.00097] | -0.00065 [-0.00206, +0.00075] |
|        1 |  0.25000 | 2.00000 | 4.00000 | 0.25000 | 1.00000 |      0.33172 |           0.28394 | 0.30783 | -0.00125 [-0.00265, +0.00021] | -0.00019 [-0.00141, +0.00102] |

- **Hybrid4** (best by mean): {'pers': 2.0, 'rate': 1.0, 'exp_4': 3.0, 'ar': 0.25, 'mf': 1.0} → main 0.33361, secondary 0.28435, mean 0.30898 → `results/submission_hybrid4.csv`
- **Hybrid5** (best with w_exp ≤ 1, conservative): {'pers': 1.0, 'rate': 0.25, 'exp_4': 1.0, 'ar': 0.25, 'mf': 1.0} → main 0.32982, secondary 0.28163, mean 0.30572 → `results/submission_hybrid5.csv`
- Refit on weeks 11-30 with the 31-42 schedule pool (6,632 shows); files validated with the Phase A checks.

Phase D2 runtime 87.4s

**Best mean MAP@5 by exposure half-life:** 4 weeks 0.30898, 8 weeks 0.30767, none 0.30371.

**Best mean MAP@5 by w_exp:** 0.5 → 0.29949, 1 → 0.30572, 2 → 0.30855, 3 → 0.30898, 4 → 0.30863.

**Interpretation.** Hybrid4 is not distinguishable from Hybrid3: the paired CIs include 0 on both splits (+0.00064 on main, +0.00021 on secondary). The validation surface is flat around w_exp 2-4. The conservative Hybrid5 (w_exp 1) is about 0.003 lower on both splits; it would only help if exposure is over-weighted on the test weeks, as the Kaggle numbers above suggest.

## 11. Phase E: coarse habit components, rank fusion, candidate set (generated by src/phase_e.py)

- **Channel exposure**: slot = channel. **Daypart exposure**: slot = (channel, weekday/weekend, 4-hour block) = 12 slots per channel. Same construction as the slot model (profile from ALL history views clipped at the target-window start, / (Σ week weights × 3600); item airtime from the target-window schedule; score = expected seconds).
- **Rank fusion (RRF)**: score = Σ_c w_c / (60 + rank_c(item)) over each component's valid items (rank 1 = highest score, ties → smaller id); valid = union of components with w > 0; fallback fill as always.
- Base = Hybrid5 without exposure: {'pers': 1.0, 'rate': 0.25, 'ar': 0.25, 'mf': 1.0}. Reference Hybrid5 (min-max, slot_4 = 1): main 0.32982, secondary 0.28163, mean 0.30572.
- Ranking: mean of main and secondary MAP@5; Δ = paired bootstrap vs Hybrid5 (1000 resamples, seed 0).

Single habit components (alone, min-max = plain ranking):

| component    |    main |   secondary |    mean |   fit_s_main |
|:-------------|--------:|------------:|--------:|-------------:|
| slot_4       | 0.31623 |     0.26773 | 0.29198 |      1.94228 |
| daypart_8    | 0.27880 |     0.23564 | 0.25722 |      1.81605 |
| daypart_None | 0.27873 |     0.23338 | 0.25605 |      1.84589 |
| daypart_4    | 0.27597 |     0.23558 | 0.25577 |      1.80819 |
| channel_8    | 0.23974 |     0.20107 | 0.22040 |      1.80837 |
| channel_4    | 0.23867 |     0.20071 | 0.21969 |      1.78154 |
| channel_None | 0.23824 |     0.20089 | 0.21957 |      1.78318 |

Hybrid grid (44 configs), top 15:

| habit        |   w_habit | fusion   |    main | d_main                        |   secondary | d_secondary                   |    mean |
|:-------------|----------:|:---------|--------:|:------------------------------|------------:|:------------------------------|--------:|
| slot_4       |   1.00000 | minmax   | 0.32982 | +0.00000 [+0.00000, +0.00000] |     0.28163 | +0.00000 [+0.00000, +0.00000] | 0.30572 |
| daypart_4    |   1.00000 | minmax   | 0.32275 | -0.00707 [-0.01141, -0.00274] |     0.27709 | -0.00454 [-0.00841, -0.00051] | 0.29992 |
| slot_4       |   0.50000 | minmax   | 0.32234 | -0.00748 [-0.01027, -0.00490] |     0.27665 | -0.00499 [-0.00731, -0.00262] | 0.29949 |
| daypart_8    |   1.00000 | minmax   | 0.32188 | -0.00794 [-0.01239, -0.00368] |     0.27596 | -0.00567 [-0.00968, -0.00139] | 0.29892 |
| daypart_4    |   0.50000 | minmax   | 0.32243 | -0.00738 [-0.01115, -0.00346] |     0.27532 | -0.00631 [-0.00966, -0.00267] | 0.29888 |
| daypart_8    |   0.50000 | minmax   | 0.32269 | -0.00713 [-0.01082, -0.00319] |     0.27412 | -0.00751 [-0.01111, -0.00391] | 0.29840 |
| daypart_None |   0.50000 | minmax   | 0.32180 | -0.00802 [-0.01185, -0.00403] |     0.27220 | -0.00943 [-0.01313, -0.00549] | 0.29700 |
| daypart_None |   1.00000 | minmax   | 0.31942 | -0.01040 [-0.01526, -0.00596] |     0.27407 | -0.00756 [-0.01192, -0.00301] | 0.29674 |
| channel_4    |   0.50000 | minmax   | 0.32010 | -0.00972 [-0.01440, -0.00490] |     0.27174 | -0.00989 [-0.01441, -0.00501] | 0.29592 |
| channel_8    |   0.50000 | minmax   | 0.31981 | -0.01001 [-0.01483, -0.00496] |     0.27175 | -0.00988 [-0.01426, -0.00472] | 0.29578 |
| channel_None |   0.50000 | minmax   | 0.31943 | -0.01039 [-0.01533, -0.00556] |     0.27121 | -0.01042 [-0.01489, -0.00514] | 0.29532 |
| daypart_4    |   0.25000 | minmax   | 0.31591 | -0.01391 [-0.01773, -0.01020] |     0.27023 | -0.01140 [-0.01501, -0.00780] | 0.29307 |
| slot_4       |   0.25000 | minmax   | 0.31499 | -0.01483 [-0.01848, -0.01156] |     0.27104 | -0.01059 [-0.01374, -0.00703] | 0.29301 |
| channel_4    |   0.25000 | minmax   | 0.31597 | -0.01385 [-0.01785, -0.00983] |     0.26974 | -0.01189 [-0.01585, -0.00764] | 0.29285 |
| channel_None |   0.25000 | minmax   | 0.31618 | -0.01363 [-0.01775, -0.00942] |     0.26924 | -0.01239 [-0.01643, -0.00796] | 0.29271 |

Min-max slot ladder and best per habit family:

| habit     |   w_habit | fusion   |    main | d_main                        |   secondary | d_secondary                   |    mean |
|:----------|----------:|:---------|--------:|:------------------------------|------------:|:------------------------------|--------:|
| none      |   0.00000 | minmax   | 0.30497 | -0.02485 [-0.02929, -0.02045] |     0.26378 | -0.01785 [-0.02195, -0.01385] | 0.28437 |
| slot_4    |   0.25000 | minmax   | 0.31499 | -0.01483 [-0.01848, -0.01156] |     0.27104 | -0.01059 [-0.01374, -0.00703] | 0.29301 |
| slot_4    |   0.50000 | minmax   | 0.32234 | -0.00748 [-0.01027, -0.00490] |     0.27665 | -0.00499 [-0.00731, -0.00262] | 0.29949 |
| slot_4    |   1.00000 | minmax   | 0.32982 | +0.00000 [+0.00000, +0.00000] |     0.28163 | +0.00000 [+0.00000, +0.00000] | 0.30572 |
| daypart_4 |   1.00000 | minmax   | 0.32275 | -0.00707 [-0.01141, -0.00274] |     0.27709 | -0.00454 [-0.00841, -0.00051] | 0.29992 |
| channel_4 |   0.50000 | minmax   | 0.32010 | -0.00972 [-0.01440, -0.00490] |     0.27174 | -0.00989 [-0.01441, -0.00501] | 0.29592 |
| slot_4    |   1.00000 | rank     | 0.28912 | -0.04070 [-0.04799, -0.03378] |     0.24082 | -0.04081 [-0.04815, -0.03415] | 0.26497 |
| daypart_4 |   1.00000 | rank     | 0.28780 | -0.04202 [-0.04981, -0.03499] |     0.24103 | -0.04060 [-0.04853, -0.03341] | 0.26441 |
| channel_4 |   1.00000 | rank     | 0.28212 | -0.04770 [-0.05596, -0.03971] |     0.23710 | -0.04453 [-0.05246, -0.03709] | 0.25961 |
| none      |   0.00000 | rank     | 0.24035 | -0.08947 [-0.09947, -0.08007] |     0.19836 | -0.08327 [-0.09305, -0.07437] | 0.21935 |

Candidate submissions (refit weeks 11-30, pool 6,632 from the 31-42 schedule, all validated; refit 24.2s):

| file                       | components                | weights                                     | fusion   |   half_life |    main |   secondary |    mean | why                      |
|:---------------------------|:--------------------------|:--------------------------------------------|:---------|------------:|--------:|------------:|--------:|:-------------------------|
| results/submission_e01.csv | pers+rate+ar+mf           | pers=1 rate=0.25 ar=0.25 mf=1               | minmax   |             | 0.30497 |     0.26378 | 0.28437 | slot ladder w=0.0        |
| results/submission_e02.csv | pers+rate+ar+mf+slot_4    | pers=1 rate=0.25 ar=0.25 mf=1 slot_4=0.25   | minmax   |           4 | 0.31499 |     0.27104 | 0.29301 | slot ladder w=0.25       |
| results/submission_e03.csv | pers+rate+ar+mf+slot_4    | pers=1 rate=0.25 ar=0.25 mf=1 slot_4=0.5    | minmax   |           4 | 0.32234 |     0.27665 | 0.29949 | slot ladder w=0.5        |
| results/submission_e04.csv | pers+rate+ar+mf+daypart_4 | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1   | minmax   |           4 | 0.32275 |     0.27709 | 0.29992 | best daypart by mean     |
| results/submission_e05.csv | pers+rate+ar+mf+channel_4 | pers=1 rate=0.25 ar=0.25 mf=1 channel_4=0.5 | minmax   |           4 | 0.3201  |     0.27174 | 0.29592 | best channel by mean     |
| results/submission_e06.csv | pers+rate+ar+mf+slot_4    | pers=1 rate=0.25 ar=0.25 mf=1 slot_4=1      | rank     |           4 | 0.28912 |     0.24082 | 0.26497 | best rank fusion by mean |
| results/submission_e07.csv | pers+rate+ar+mf+slot_4    | pers=1 rate=0.25 ar=0.25 mf=1 slot_4=1      | minmax   |           4 | 0.32982 |     0.28163 | 0.30572 | overall best by mean     |

**Selection plan:** the final pick among these files is made by Kaggle PUBLIC score; the private leaderboard stays held out. These validation numbers were not tuned on Kaggle scores.

Phase E runtime 106.3s

**Notes**
- `submission_e07.csv` (overall best by mean) is byte-identical to `submission_hybrid5.csv`, which confirms that the pipeline reproduces Hybrid5 exactly. It does not need to be submitted again.
- Mean item overlap of each file with Hybrid5: e01 0.818, e02 0.873, e03 0.922, e04 0.845, e05 0.821, e06 0.768.
- **On validation, coarser habits do not beat the exact slot.** Taken alone: slot 0.292 > daypart 0.257 > channel 0.220 (mean of the two splits). In the hybrid at its best weight: daypart 0.29992 and channel 0.29592, both below Hybrid5's 0.30572, with paired CIs excluding 0. Validation cannot test the autumn-grid hypothesis, because its target windows share the spring/summer grid. That is why the coarse variants are offered as public-leaderboard candidates.
- **RRF (k = 60) is much worse than min-max fusion** (best mean 0.26497 vs 0.30572). With k = 60, a component's contributions barely change across the top ranks (1/61 vs 1/65), so the order of a strong component like personal history gets diluted by the full rankings from MF and exposure. MF ranks every pool item.

## 12. Phase F: several habit components at once (src/phase_f.py)

**Kaggle results reported by the user from their manual submissions (private / public; not computed in this repo):**

| submission | private | public |
|---|---|---|
| e04 (base + daypart_4 1) | 0.28302 | 0.31297 |
| e05 (base + channel_4 0.5) | 0.27870 | 0.31350 |
| Hybrid5 (base + slot_4 1) | 0.27402 | 0.30492 |
| e03 (base + slot_4 0.5) | 0.27389 | 0.30468 |
| e02 (base + slot_4 0.25) | 0.27144 | 0.30311 |
| e01 (base only) | 0.26544 | 0.29997 |
| e06 (RRF) | 0.26148 | 0.29088 |

Coarse habit components transfer to the test weeks better than exact hour slots. This is the opposite of the validation ranking, consistent with the autumn grid differing from the spring/summer grid that both validation windows share.

**Configs.**
- All configs use base = pers 1, rate 0.25, AR 0.25, MF 1, with per-user min-max fusion. The habit components below are added to it.
- The 8 configs were specified by the user. None were tuned here. Validation scores are reported for the record only, and they are *not* used for selection.
- Each file was refit on weeks 11-30 with the 6,632-show pool from the 31-42 schedule and validated with the Phase A checks.
- `overlap_with_e04` = mean share of a row's 5 items that also appear in e04.

| file                       | habits                                   | weights                                                            | fusion   |    main |   secondary |    mean |   overlap_with_e04 |
|:---------------------------|:-----------------------------------------|:-------------------------------------------------------------------|:---------|--------:|------------:|--------:|-------------------:|
| results/submission_f01.csv | daypart_4 1.5                            | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1.5                        | minmax   | 0.31941 |     0.27387 | 0.29664 |              0.932 |
| results/submission_f02.csv | daypart_4 2                              | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=2                          | minmax   | 0.31847 |     0.26955 | 0.29401 |              0.891 |
| results/submission_f03.csv | daypart_4 1 + channel_4 0.5              | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1 channel_4=0.5            | minmax   | 0.32297 |     0.27591 | 0.29944 |              0.908 |
| results/submission_f04.csv | daypart_4 1 + channel_4 1                | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1 channel_4=1              | minmax   | 0.31814 |     0.26884 | 0.29349 |              0.845 |
| results/submission_f05.csv | daypart_4 1 + slot_4 0.5                 | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1 slot_4=0.5               | minmax   | 0.32662 |     0.28125 | 0.30394 |              0.932 |
| results/submission_f06.csv | daypart_None 1                           | pers=1 rate=0.25 ar=0.25 mf=1 daypart_None=1                       | minmax   | 0.31942 |     0.27407 | 0.29674 |              0.876 |
| results/submission_f07.csv | daypart_4 1 + channel_4 0.5 + slot_4 0.5 | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1 channel_4=0.5 slot_4=0.5 | minmax   | 0.32803 |     0.28128 | 0.30465 |              0.894 |
| results/submission_f08.csv | channel_4 1                              | pers=1 rate=0.25 ar=0.25 mf=1 channel_4=1                          | minmax   | 0.31428 |     0.26517 | 0.28972 |              0.803 |

Phase F runtime 62.5s

**Consistency check.** f06 (daypart_None 1) reproduces the Phase E value for the same config: main 0.31942.

**Validation vs Kaggle.**
- On validation, configs that include the exact slot (f05, f07) rank highest. On Kaggle so far, the slot model has transferred worst.
- The validation ordering is therefore not informative for this choice.

**Selection rule.** The final submission is chosen by Kaggle PUBLIC score among all candidate files; the private leaderboard stays held out. Choosing among several files by public score carries some risk of overfitting the public split, so a large public gap matters more than a small one.

## 13. Phase G: title families (src/family.py, src/phase_g.py)

**Kaggle results reported by the user from their manual submissions (private / public; not computed in this repo):**

| submission | private | public |
|---|---|---|
| f03 | 0.28638 (best private) | 0.31287 |
| f07 | 0.28433 | 0.31230 |
| f04 | 0.28317 | 0.31000 |
| f01 | 0.28207 | 0.30944 |
| f05 | 0.28123 | 0.31074 |
| f06 | 0.28058 | 0.31374 (best public) |
| f08 | 0.27961 | 0.31216 |
| f02 | 0.27933 | 0.30586 |

**Hypothesis tested.** New seasons or cycles get new tv_show_ids, so a show's earlier ids could be linked to its new ones through a title family.

**Family key (schedule metadata only).**
- Lower-case; keep the type prefix (т/с, х/ф, м/с, м/ф, ...); strip quotes.
- Cut at the *first season/episode/series marker* (", N сезон", "N эп.", "N с.", "N-M с.", "серия N", ...). Drop the remaining numbers and punctuation.
- If only the prefix is left, keep the digits.
- Each id is mapped through its most frequent title.
- **Deviation from the requested "drop the sub-title after the first comma".** The first-comma rule was run first and produced obvious wrong merges: films with commas in the title ('Я, робот' + 'Я, Алекс Кросс' → `х/ф я`; 'Дом, …' ×8), all numeric film titles ('2012', '13', '1+1' → `х/ф`), and an umbrella `сказки` family. So the cut point was moved to the first marker. Under the first-comma key, 126 of the new 31-42 ids joined a history family; under the final key, 106 do.
- **Remaining cosmetic issue:** "1 и 2 с." leaves a trailing "и" in a few series keys (e.g. `т/с тайны следствия и`). It is applied consistently, so it rarely splits a family.

### Family statistics

- labelled ids: 9,671; families: 9,049; families with >= 2 ids: 467 covering 1,089 ids

20 largest families:

| family                                           |   n_ids | categories   | example titles                                                                                                                 |
|:-------------------------------------------------|--------:|:-------------|:-------------------------------------------------------------------------------------------------------------------------------|
| х/ф три в одном                                  |       6 | Фильмы       | Х/ф 'Три в одном', 1 с. | Х/ф 'Три в одном 2', 1 с. | Х/ф 'Три в одном 3', 1 с.                                                |
| т/с солдаты                                      |       6 | Сериалы      | Т/с 'Солдаты', 1-3 с. | Т/с 'Солдаты 3', 1 с. | Т/с 'Солдаты 4', 1-3 с.                                                        |
| х/ф стрелок                                      |       6 | Фильмы       | Х/ф 'Стрелок', 1 ч. | Х/ф 'Стрелок'. | Х/ф 'Стрелок'.                                                                          |
| х/ф форсаж                                       |       6 | Фильмы       | Х/ф 'Форсаж'. | Х/ф 'Форсаж 4'. | Х/ф 'Форсаж 5'.                                                                              |
| х/ф такси                                        |       5 | Фильмы       | Х/ф 'Такси 2'. | Х/ф 'Такси'. | Х/ф 'Такси 3'.                                                                                 |
| х/ф заложница                                    |       5 | Фильмы       | Х/ф 'Заложница'. | Х/ф 'Заложница'. | Х/ф 'Заложница 2'.                                                                       |
| х/ф снайпер                                      |       5 | Фильмы       | Х/ф 'Снайпер'. | Х/ф 'Снайпер 2'. | Х/ф 'Снайпер 3', 1 ч.                                                                      |
| х/ф рокки                                        |       5 | Фильмы       | Х/ф 'Рокки'. | Х/ф 'Рокки 2'. | Х/ф 'Рокки 3'.                                                                                 |
| х/ф женская логика                               |       5 | Фильмы       | Х/ф 'Женская логика'. | Х/ф 'Женская логика 2'. | Х/ф 'Женская логика 3'.                                                      |
| х/ф пункт назначения                             |       5 | Фильмы       | Х/ф 'Пункт назначения'. | Х/ф 'Пункт назначения 2'. | Х/ф 'Пункт назначения 3'.                                                |
| т/с агент национальной безопасности и            |       5 | Сериалы      | Т/с 'Агент национальной безопасности', 1 | Т/с 'Агент национальной безопасности 2', | Т/с 'Агент национальной безопасности 3', |
| х/ф бетховен                                     |       5 | Фильмы       | Х/ф 'Бетховен'. | Х/ф 'Бетховен 2'. | Х/ф 'Бетховен 3'.                                                                        |
| х/ф крепкий орешек                               |       5 | Фильмы       | Х/ф 'Крепкий орешек 2'. | Х/ф 'Крепкий орешек'. | Х/ф 'Крепкий орешек'.                                                        |
| т/с тайны следствия и                            |       5 | Сериалы      | Т/с 'Тайны следствия', 1 и 2 с. | Т/с 'Тайны следствия 4', 1 и 2 с. | Т/с 'Тайны следствия 5', 1 и 2 с.                        |
| х/ф робокоп возвращается                         |       4 | Фильмы       | Х/ф 'Робокоп возвращается 4'. | Х/ф 'Робокоп возвращается'. | Х/ф 'Робокоп возвращается 3'.                                    |
| т/с даша васильева любительница частного сыска и |       4 | Сериалы      | Т/с 'Даша Васильева. Любительница частно | Т/с 'Даша Васильева. Любительница частно | Т/с 'Даша Васильева. Любительница частно |
| х/ф морской пехотинец                            |       4 | Фильмы       | Х/ф 'Морской пехотинец'. | Х/ф 'Морской пехотинец 2'. | Х/ф 'Морской пехотинец 3'.                                             |
| х/ф час пик                                      |       4 | Фильмы       | Х/ф 'Час пик'. | Х/ф 'Час пик 2'. | Х/ф 'Час пик 3'.                                                                           |
| х/ф смертельное оружие                           |       4 | Фильмы       | Х/ф 'Смертельное оружие 2'. | Х/ф 'Смертельное оружие'. | Х/ф 'Смертельное оружие 3'.                                          |
| х/ф робокоп                                      |       4 | Фильмы       | Х/ф 'Робокоп'. | Х/ф 'Робокоп 2'. | Х/ф 'Робокоп 3'.                                                                           |

- labelled ids airing in weeks 31-42 that never appear in the 11-30 schedules: 1,435; of these, joining a family seen in 11-30: **106** (7.4%)

15 random merged ids (seed 0):

|   tv_show_id | title                             | family                      | other ids' titles                                                   |
|-------------:|:----------------------------------|:----------------------------|:--------------------------------------------------------------------|
|     90048220 | Х/ф 'Плохие парни'.               | х/ф плохие парни            | Х/ф 'Плохие парни 2'.                                               |
|     90075067 | Х/ф 'Срочная доставка'.           | х/ф срочная доставка        | Х/ф 'Срочная доставка'.                                             |
|     90082268 | Х/ф 'Игра'.                       | х/ф игра                    | Х/ф 'Игра'. | Х/ф 'Игра'.                                           |
|     90074437 | Х/ф 'Шпион'.                      | х/ф шпион                   | Х/ф 'Шпион'.                                                        |
|     90050203 | Х/ф 'Изгой'.                      | х/ф изгой                   | Х/ф 'Изгой'.                                                        |
|     90079958 | Х/ф 'Заложница'.                  | х/ф заложница               | Х/ф 'Заложница'. | Х/ф 'Заложница'.                                 |
|     90080767 | Х/ф 'Провинциалка'.               | х/ф провинциалка            | Х/ф 'Провинциалка'.                                                 |
|      1000955 | Т/с 'Солдаты 8', 1 и 2 с.         | т/с солдаты и               | Т/с 'Солдаты 2', 1 и 2 с.                                           |
|     90070678 | Х/ф 'Любовь в большом городе 2'.  | х/ф любовь в большом городе | Х/ф 'Любовь в большом городе'. | Х/ф 'Любовь в большом городе 3'.   |
|     90070809 | Х/ф 'Дело 39'.                    | х/ф дело                    | Х/ф 'Дело №306'.                                                    |
|    120078240 | М/ф 'Как приручить дракона 2'.    | м/ф как приручить дракона   | М/ф 'Как приручить дракона'. | М/ф 'Как приручить дракона 3'.       |
|     10002976 | Т/с 'Женский доктор 2', 32 с.     | т/с женский доктор          | Т/с 'Женский доктор', 7 с. | Т/с 'Женский доктор 3', 18 с.          |
|      1000433 | Т/с 'Тайны следствия 4', 1 и 2 с. | т/с тайны следствия и       | Т/с 'Тайны следствия', 1 и 2 с. | Т/с 'Тайны следствия 5', 1 и 2 с. |
|     90039050 | Х/ф 'Жизнь прекрасна'.            | х/ф жизнь прекрасна         | Х/ф 'Жизнь прекрасна', 1-3 с.                                       |
|     90075497 | Х/ф 'Прекрасные создания'.        | х/ф прекрасные создания     | Х/ф 'Прекрасные создания', 1-4 с.                                   |

- possible wrong merges: families with >= 2 ids mixing categories: 0 (e.g. ); un-prefixed one-word keys with >= 2 ids: 0 (e.g. )

Validation target items reachable through families:

- main: target items 6,221; ids not airing in history weeks 513 (8.2%); of these joining a history family 54; joining a family THE USER watched in history 6
- secondary: target items 6,204; ids not airing in history weeks 481 (7.8%); of these joining a history family 30; joining a family THE USER watched in history 1

Results (MAP@5; Δ = paired bootstrap vs f03, 1000 resamples, seed 0):

| config                              | fallback   |    main | Δ main vs f03                 |   secondary | Δ secondary vs f03            |    mean |
|:------------------------------------|:-----------|--------:|:------------------------------|------------:|:------------------------------|--------:|
| f03 + famR 1 + family fallback      | family     | 0.32889 | +0.00593 [+0.00233, +0.00924] |     0.27951 | +0.00361 [+0.00086, +0.00655] | 0.30420 |
| f03 + famR 1                        | id         | 0.32889 | +0.00593 [+0.00233, +0.00924] |     0.27951 | +0.00361 [+0.00086, +0.00655] | 0.30420 |
| ablation: f03 with id-rate 1.25     | id         | 0.32864 | +0.00567 [+0.00211, +0.00895] |     0.27941 | +0.00350 [+0.00076, +0.00649] | 0.30402 |
| f03 + famP 1                        | id         | 0.32594 | +0.00297 [-0.00036, +0.00623] |     0.27978 | +0.00388 [+0.00085, +0.00672] | 0.30286 |
| ablation: f03 with pers 2           | id         | 0.32598 | +0.00301 [-0.00033, +0.00623] |     0.27963 | +0.00373 [+0.00074, +0.00658] | 0.30281 |
| f03 + famR 0.5                      | id         | 0.32647 | +0.00350 [+0.00115, +0.00567] |     0.27853 | +0.00263 [+0.00052, +0.00479] | 0.30250 |
| f03 + famP 2                        | id         | 0.32583 | +0.00286 [-0.00165, +0.00719] |     0.27848 | +0.00258 [-0.00122, +0.00644] | 0.30216 |
| f03 + famP 0.5                      | id         | 0.32577 | +0.00280 [+0.00057, +0.00498] |     0.27815 | +0.00224 [+0.00031, +0.00423] | 0.30196 |
| f03 + famR 2                        | id         | 0.32784 | +0.00487 [+0.00015, +0.00946] |     0.27594 | +0.00003 [-0.00337, +0.00397] | 0.30189 |
| f03 + family fallback               | family     | 0.32297 | +0.00000 [+0.00000, +0.00000] |     0.27591 | +0.00000 [+0.00000, +0.00000] | 0.29944 |
| f03 (reference)                     | id         | 0.32297 | +0.00000 [+0.00000, +0.00000] |     0.27591 | +0.00000 [+0.00000, +0.00000] | 0.29944 |
| f03, pers -> famP                   | id         | 0.32303 | +0.00006 [-0.00014, +0.00037] |     0.27582 | -0.00009 [-0.00027, +0.00000] | 0.29942 |
| f03, pers -> famP + family fallback | family     | 0.32303 | +0.00006 [-0.00014, +0.00037] |     0.27582 | -0.00009 [-0.00027, +0.00000] | 0.29942 |
| f03, pers -> famR                   | id         | 0.32086 | -0.00211 [-0.00526, +0.00083] |     0.27212 | -0.00378 [-0.00673, -0.00104] | 0.29649 |
| pers alone                          | id         | 0.29708 | -0.02588 [-0.03252, -0.01949] |     0.26248 | -0.01343 [-0.01952, -0.00730] | 0.27978 |
| famP alone                          | id         | 0.29576 | -0.02721 [-0.03387, -0.02072] |     0.26182 | -0.01408 [-0.02016, -0.00774] | 0.27879 |
| famR alone                          | id         | 0.27922 | -0.04375 [-0.05077, -0.03586] |     0.22721 | -0.04870 [-0.05544, -0.04164] | 0.25321 |

Candidate files (refit weeks 11-30, pool 6,632 from the 31-42 schedule, validated; 22.9s):

| file                       | config                          | fallback   | weights                                                        |    main |   secondary |    mean |   overlap_with_f03 | why                                   |
|:---------------------------|:--------------------------------|:-----------|:---------------------------------------------------------------|--------:|------------:|--------:|-------------------:|:--------------------------------------|
| results/submission_g01.csv | f03 + famR 1                    | id         | pers=1 rate=0.25 ar=0.25 mf=1 daypart_4=1 channel_4=0.5 famR=1 | 0.32889 |     0.27951 | 0.3042  |              0.902 | f03 + family at best weight by mean   |
| results/submission_g02.csv | f03, pers -> famP               | id         | rate=0.25 ar=0.25 mf=1 daypart_4=1 channel_4=0.5 famP=1        | 0.32303 |     0.27582 | 0.29942 |              0.998 | family replaces pers (best by mean)   |
| results/submission_g03.csv | ablation: f03 with id-rate 1.25 | id         | pers=1 rate=1.25 ar=0.25 mf=1 daypart_4=1 channel_4=0.5        | 0.32864 |     0.27941 | 0.30402 |              0.902 | ablation: same gain without families? |

- `f03 + famR 1 + family fallback` (family fallback; best family-fallback variant by mean) gives predictions identical to results/submission_g01.csv on the final refit -> not written.

Phase G runtime 72.0s


**Interpretation.**
- **The bridging hypothesis is largely refuted.** Only 106 of the 1,435 new 31-42 ids (7.4%) join a family seen in weeks 11-30. In validation, only 6 (main) and 1 (secondary) target items are new ids from a family *the same user* watched. Most largest families are film franchises and long-running series; news bulletins never merge with other shows.
- **The f03 + family-rate gain does not come from families.** It is +0.0059 on main and +0.0036 on secondary, significant on both. But giving the plain per-id rate component weight 1.25 instead of 0.25 reproduces it: 0.32864 / 0.27941 vs 0.32889 / 0.27951. The gain is the rate signal getting more weight.
- **The family-aware fallback never changes predictions.** With MF in the hybrid, every pool item is valid for users with history, so the fallback is never reached. Its final file was identical to g01 and was not written.
- **g02 (family replaces pers) is a near-duplicate of f03**, sharing 99.8% of items.
- **g01 vs g03 are nearly identical on the final refit:** they share 0.998 of their items, and 97.9% of rows are identical. The family component adds essentially nothing beyond more rate weight, so submitting both is unlikely to be informative.
- **Selection rule (unchanged):** pick by Kaggle public score; private stays held out.
