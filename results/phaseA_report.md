# Phase A report: sweet.tv TV program recommender

All numbers below come from scripts in `src/` run on 2026-10-24 (environment: `results/environment.md`).
Full generated outputs: `results/audit.md`, `results/join_report.md`, `results/baselines.md`,
`results/baselines.csv`, `results/fig_overlap.png`.

Reproduce (from `src/`, using `../.venv/bin/python`): `audit.py` → `join_report.py` (builds the join
cache) → `baselines.py` → `make_submission.py`; tests: `.venv/bin/python -m pytest tests`.

## 1. Data facts

| file | rows | time range | ISO weeks (of start) |
|---|---|---|---|
| dataset11-30.csv (views) | 5,821,987 (17 exact duplicates) | 2020-03-09 00:00:02 .. 2020-07-26 23:58:42 | 11..30 |
| export_arh_11-20 | 1,074,235 | 2020-03-08 23:00 .. 2020-05-18 11:45 | 10..21 (86 rows in wk 10, 2,431 in wk 21) |
| export_arh_21-30 | 1,074,394 | 2020-05-18 00:00 .. 2020-07-27 11:55 | 21..31 (2,429 rows in wk 31) |
| export_arh_31-42 | 1,295,378 | 2020-07-27 00:00 .. 2020-10-19 11:55 | 31..43 (2,399 rows in wk 43) |
| submission.csv | 1,260 users, header `user_id,tv_show_id`, LF, **no trailing newline**, ids `0 0 0 0 0` | | |

- **Timestamp formats.** The log uses `%Y-%m-%d %H:%M:%S` and the schedules use `%d.%m.%Y %H:%M:%S`. Both were parsed with an explicit `format=`, so any row that did not match would have raised an error.
- **ISO weeks.** The ISO week of every view start matches its date: week 11 starts on Monday 2020-03-09 and week 30 ends on Sunday 2020-07-26. The 31-42 schedule starts on Monday 2020-07-27, which is ISO week 31. Each schedule file spills about half a day into the neighbouring week (broadcast day). The files do not duplicate each other: they share no (channel, start) pair, and only 1 pair of consecutive airings overlaps across files.
- **Time alignment of log vs schedule.** I shifted view times by s and counted the share of view starts within ±30 s of an airing start on the same channel. The share is highest at s = 0: 1.81% at 0, against 1.78% at −600 s, 1.73% at −3600 s and 1.67% at +3600 s. This holds both before and after the 2020-03-29 DST change. The evidence is consistent but weak, so no shift is applied.
- **Log.** 4,914 users, 202 channels. `duraton` = stop−start in 99.998% of rows; the minimum is 61 s, so the log contains no views shorter than a minute. Median view is 396 s, mean 1,480 s, and 0.96% of views are longer than 4 h. 17,368 views cross the Sunday→Monday boundary; their week is taken from the start time.
- **Activity drops sharply in weeks 20-27.** Weeks 11-19 have about 400k rows per week, weeks 21-27 have 105-193k (users per week fall to 2,406 in week 26), and weeks 28-30 recover to about 270k.
- **Schedules.** 400-404 channels. tv_show_id ≠ 0 on 10.76% / 11.31% / 11.18% of rows, which is 16-17% of airtime. Distinct labelled shows: 5,847 / 6,363 / 6,636. 5,199 of the 6,636 shows in 31-42 also appear in 11-30. Median airing length is 1,500 s for all rows and 2,700 s for labelled rows. 4,351 ids carry more than one title (episode titles).
- **Submission users.** All 1,260 have log rows (37.3% of the log). Median activity is 20 of 20 weeks active and 536 h watched. 1,258 of 1,260 have at least one labelled interaction.

## 2. Join rule

A view (user, channel c, [start, stop]) is matched to every airing on the **same channel c** whose interval [start_time, start_time + duration] overlaps it. Overlap = min(stops) − max(starts), counted only when it is > 0. Rows are aggregated to (user, airing, ISO week of the view start), and the summed overlap is capped at the airing duration. The implementation (`src/join.py`) uses vectorised per-channel `searchsorted` and takes 4.3 s for the full log, producing 6,330,721 rows.

Share of the 2,394,224 viewing hours:

| | share |
|---|---|
| labelled airtime (tv_show_id ≠ 0) | 52.36% |
| unlabelled airtime (id 0) | 40.68% |
| no airing (channel absent from schedule / gaps) | 6.96% |

For submission users only, the shares are 50.89% labelled and 42.05% unlabelled.

**Hand-checkable example** (`join_report.md`). User 8603729495312378564 watched channel 3 from 2020-04-25 11:34:27 to 12:33:49 (3,562 s). Airing 10:00-12:05 ('Джуниор', id 90014551) overlaps from 11:34:27 to 12:05:00, which is 1,833 s. Airing 12:05-14:05 ('Кара небесная', id 90014056) overlaps from 12:05:00 to 12:33:49, which is 1,729 s. The manual results equal the join output, and 1,833 + 1,729 = 3,562 = duraton.

## 3. "Watched" threshold

Overlap per user × labelled airing (3,017,105 pairs):

| overlap | share of airings | share of seconds |
|---|---|---|
| < 60 s | 2.5% | 0.05% |
| 60-300 s | 28.5% | 2.8% |
| 300-900 s | 23.8% | 8.8% |
| ≥ 900 s | 45.2% | 88.4% |

- **Quantiles of overlap:** median 711 s, q25 213 s, q75 1,989 s.
- **Quantiles of overlap/duration:** median 0.195. 27% of airings have a fraction below 0.05, and 17.5% have ≥ 0.9; those ≥ 0.9 airings hold 34.8% of the seconds.

The figure shows a step at 60 s, which is the log's minimum view length. Below 60 s, overlaps come only from views that start or end at an airing boundary. From 60 to 300 s there is a long flat zapping band: it contains many airings but little time.

**Proposal: watched = overlap ≥ 300 s, with no fraction rule.** This removes 31% of airings, which are mostly zapping, but only 2.9% of viewing seconds.

I rejected a fraction rule for three reasons:
- frac ≥ 0.5 drops 35% of the seconds.
- frac ≥ 0.25 drops 16.5% of the seconds.
- Both penalise long films and marathon blocks even when the user watched them for hours.

`min_sec` and `min_frac` are parameters in `join.watched()`.

**The target uses ≥ 300 s.** On validation, the history threshold made little difference: 0 s ≈ 300 s, and both beat 900 s (table below). The ≥ 300 s threshold also leaves the target ranking almost unchanged, because ranking is by seconds and short overlaps carry almost no seconds.

## 4. Validation protocol (identical for every method, now and later)

- **Splits:**
  - main: history weeks 11-20 → target weeks 21-30
  - secondary: history 11-24 → target 25-30
- **Target (our hypothesis of the competition's "preference"):** per user, the top 5 labelled shows by watched seconds in the target window (airings with overlap ≥ 300 s). Ties are broken by tv_show_id.
- **Eval users:** submission users with a non-empty target. That is 1,246 of 1,260 on main (1,243 of them have 5 target items) and 1,244 of 1,260 on secondary.
- **Candidate pool:** labelled shows with an airing that *starts* in a target-window week, taken from the schedule only. That is 6,376 shows on main, 5,013 on secondary and 6,632 for the final 31-42 pool. No target item falls outside the pool (0 of 818 on main, 0 of 855 on secondary).
- **Repeats are allowed.** Shows the user already watched can be recommended. This matters: 66.4% (main) and 62.3% (secondary) of eval users' target items were already watched by that user in history (≥ 300 s).
- **Cold-start fallback:** pool popularity by watched seconds over all history users (`b_pop_pool`). It also fills every list to 5 items.
- **Metric:** MAP@5 with Kaggle semantics (`src/metrics.py`): AP@5 = Σ P(k)·rel(k) / min(|truth|, 5), averaged over eval users, and a duplicate prediction is not credited twice. 8 unit tests on hand-made cases pass (`tests/test_metrics.py`).
- **Training data:** models use only history-week viewing. The target-window schedule is used only to build the pool.

## 5. Reference results: MAP@5

The target threshold is 300 s. Columns are the history watch threshold. Popularity is computed over all 4,914 log users.

| method | main ≥0s | main ≥300s | main ≥900s | sec ≥0s | sec ≥300s | sec ≥900s |
|---|---|---|---|---|---|---|
| (a) pop_all (no pool restriction) | 0.17685 | 0.17913 | 0.17913 | 0.15295 | 0.15295 | 0.15563 |
| (b) pop_pool | 0.17685 | 0.17913 | 0.17913 | 0.15295 | 0.15295 | 0.15563 |
| (b2) pop_pool by #users | 0.17409 | 0.17752 | 0.17625 | 0.15122 | 0.15122 | 0.15029 |
| (c) pop_pool, half-life 1 wk | 0.18256 | **0.18674** | 0.18674 | 0.15799 | 0.15799 | 0.15445 |
| (c) pop_pool, half-life 2 wk | 0.18256 | 0.18256 | 0.18256 | 0.15799 | 0.15799 | 0.15799 |
| (c) pop_pool, half-life 4 wk | 0.17913 | 0.17913 | 0.18256 | 0.15563 | 0.15563 | 0.15799 |
| (d) personal (sum of seconds) | **0.29708** | 0.29497 | 0.28301 | 0.26248 | 0.26084 | 0.25350 |
| (d) personal, half-life 1 wk | 0.22301 | 0.22112 | 0.20384 | 0.18866 | 0.18622 | 0.17069 |
| (d) personal, half-life 2 wk | 0.27155 | 0.27052 | 0.25398 | 0.23633 | 0.23382 | 0.21969 |
| (d) personal, half-life 4 wk | 0.29608 | 0.29456 | 0.28026 | **0.26299** | 0.26112 | 0.24926 |

Sensitivity to the target threshold, with history threshold 0 s:

| | target ≥0s | ≥300s | ≥900s |
|---|---|---|---|
| main b_pop_pool | 0.18169 | 0.17685 | 0.15723 |
| main c_pop_pool_hl1 | 0.18699 | 0.18256 | 0.16317 |
| main d_personal | 0.30139 | 0.29708 | 0.27647 |
| secondary b_pop_pool | 0.15530 | 0.15295 | 0.12872 |
| secondary c_pop_pool_hl1 | 0.16033 | 0.15799 | 0.13524 |
| secondary d_personal | 0.26570 | 0.26248 | 0.23746 |

The ranking of methods is the same under every target threshold.

Runtime per fit+predict is ≤ 0.1 s for every reference. The whole `baselines.py` run takes 13.5 s, and the join takes 4.3 s.

**Observations**
- (a) and (b) are identical: the 5 most-watched shows overall all air in the target window. On the final run (weeks 11-30) the fallback top 5 is 3 news programmes (ТСН, Факты. День, Сегодня. Итоги), the travel series 'Мир наизнанку' and the morning show 'Завтрак с 1+1'.
- Recency helps popularity a little (+0.008 on main, +0.005 on secondary) but hurts the personal model. Personal taste is stable over 10-14 weeks, and averaging over all of it beats looking only at the last weeks.
- The personal baseline is about 1.7× pop_pool on both splits.

## 6. Baseline submission

- **Selected method:** (d) `d_personal` with history threshold 0 s. It is the best on the main split. On secondary it is within 0.0005 of half-life 4, which wins there.
- **Training:** retrained on weeks 11-30 (all labelled overlap). The pool is the labelled shows starting in weeks 31-42 of `export_arh_31-42` (6,632 shows). The fallback is pool popularity over 11-30: 2400480, 240081, 6500479, 2400508, 20088.
- **Coverage:** 1,258 of 1,260 users have in-pool history. The other 2 get the fallback.
- **Output:** `results/submission_baseline.csv` has the template header, 1,260 rows in template order, LF line endings and no trailing newline. Each row has 5 distinct non-zero ids, all in the pool. All of this is checked by `make_submission.validate()`. It has **not** been submitted to Kaggle.

## 7. Assumptions

- The competition's "preference" is the top 5 shows by watched seconds in weeks 31-42, with an implied threshold. The real definition is unpublished. It might count airings or days, or normalise by programme length.
- Same channel plus time overlap is the correct join, with no time-zone shift (supported only weakly, §1).
- The week of a view is the ISO week of its start time.
- Unlabelled airtime (id 0) is not an item and is ignored, even though it holds 41% of viewing time.

## 8. Risks

- **Target definition mismatch.** Ranking by seconds favours long formats (films, marathons, news blocks). If Kaggle ranks by count of airings watched, our validation could point the wrong way. Both choices are parameters, and a count-based target is worth checking as a sensitivity in Phase B.
- **Distribution shift.** Weeks 20-27 have much less activity. In addition, 31-42 is late summer to autumn, when a new TV season starts: 1,437 of the 6,636 shows airing in 31-42 never appear in the 11-30 schedules, and personal history cannot recommend them. Validation windows don't fully reflect this, so the 31-42 score may be lower than validation.
- **Horizon mismatch.** Validation horizons are 10 and 6 weeks; the test horizon is 12.
- **Time alignment.** Evidence for no shift is weak (§1). A wrong offset would corrupt the join at every programme boundary.
- **Data quirks.** 6.96% of viewing time hits no airing, 10 log channels are missing from at least one schedule, and 17 duplicate rows were dropped. 2 of the 1,260 users get pure fallback.
- **Small differences.** Differences of about 0.002 between variants are within noise: about 1,250 users and no confidence intervals yet. A bootstrap over users is planned for Phase B.
