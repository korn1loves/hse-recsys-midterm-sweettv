"""Phase L: probing the target definition with personal-history rankings.

Airing-level table: one row per (user, labelled airing) with overlap seconds (capped at the airing
duration), week of the view (min over the user's views of that airing), airing start date.
Ranking rules (history weeks; pool = window's labelled pool; ties -> total seconds, then smaller id):
  l01 sec      : total overlap seconds over all airings (= Phase A personal history, ties by id only)
  l02 airings  : number of watched airings (overlap >= 300 s)
  l03 days     : number of distinct airing dates with a watched airing
  l04 weeks    : number of distinct weeks with a watched airing
  l05 episodes : sum over all airings of min(overlap / duration, 1)
Target definitions T1..T5 = the same five quantities over WATCHED airings (>= 300 s) in the target
window, top-5 per user (ties -> seconds, then id); T1 is the existing target.
Fallback: Phase A pool popularity by seconds. Writes results/phaseL.md, results/phaseL_table.csv,
results/phaseL_manifest.csv, results/submission_l01..l05.csv.
"""
import time

import numpy as np
import pandas as pd

import baselines as B
import data
import join
import phase_d as PD
from metrics import mapk

RES = data.ROOT / "results"
RULES = ["sec", "airings", "days", "weeks", "episodes"]
NAMES = {"sec": "total seconds", "airings": "# watched airings", "days": "# distinct days",
         "weeks": "# distinct weeks", "episodes": "episode-equivalents"}
WATCH = 300


def airing_table(itx):
    lab = itx[itx.tv_show_id != 0]
    a = lab.groupby(["user_id", "airing_id"], as_index=False).agg(
        overlap=("overlap_s", "sum"), week=("week", "min"), duration=("duration", "first"),
        tv_show_id=("tv_show_id", "first"), start=("start_time", "first"))
    a["overlap"] = np.minimum(a.overlap, a.duration)
    a["day"] = a.start.dt.normalize()
    a["epi"] = np.minimum(a.overlap / a.duration.clip(lower=1), 1.0)
    return a


def scores(a, weeks, rule, target=False):
    """(user, show) -> score and total seconds (tie-break) for one rule."""
    d = a[a.week.isin(list(weeks))]
    if target:
        d = d[d.overlap >= WATCH]
    sec = d.groupby(["user_id", "tv_show_id"]).overlap.sum().rename("sec")
    w = d[d.overlap >= WATCH]
    if rule == "sec":
        s = sec.rename("score")
    elif rule == "airings":
        s = w.groupby(["user_id", "tv_show_id"]).size().rename("score")
    elif rule == "days":
        s = w.groupby(["user_id", "tv_show_id"]).day.nunique().rename("score")
    elif rule == "weeks":
        s = w.groupby(["user_id", "tv_show_id"]).week.nunique().rename("score")
    else:
        s = d.groupby(["user_id", "tv_show_id"]).epi.sum().rename("score")
    out = pd.concat([s, sec], axis=1).dropna(subset=["score"]).reset_index()
    return out[out.score > 0]


def top_lists(df, users, pool=None, tie_sec=True):
    if pool is not None:
        df = df[df.tv_show_id.isin(pool)]
    df = df[df.user_id.isin(users)]
    keys, asc = (["user_id", "score", "sec", "tv_show_id"], [True, False, False, True]) if tie_sec else \
        (["user_id", "score", "tv_show_id"], [True, False, True])
    df = df.sort_values(keys, ascending=asc)
    return {u: g.tv_show_id.head(B.K).tolist() for u, g in df.groupby("user_id")}


def recommend(a, hw, pool, users, rule, fallback):
    per = top_lists(scores(a, hw, rule), users, pool, tie_sec=(rule != "sec"))
    return {u: B.fill(per.get(u, []), fallback) for u in users}


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    a = airing_table(itx)

    rows = []
    for split, (hw, tw) in B.SPLITS.items():
        sp = B.make_split(split, itx, 0, users)
        fb = B.popularity(sp)
        targets = {t: top_lists(scores(a, tw, t, target=True), users) for t in RULES}
        preds = {r: recommend(a, hw, sp.pool, users, r, fb) for r in RULES}
        for r in RULES:
            B.check_preds(preds[r], users, sp.pool)
            row = {"split": split, "ranking": f"l0{RULES.index(r) + 1} {NAMES[r]}"}
            for t in RULES:
                row[f"T{RULES.index(t) + 1} {NAMES[t]}"] = mapk(targets[t], preds[r])
            rows.append(row)
        same_t1 = targets["sec"] == sp.target
        print(split, "T1 equals existing target:", same_t1, "eval users", {t: len(v) for t, v in targets.items()})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "phaseL_table.csv", index=False)

    # final files
    hist = join.user_show_week(join.watched(itx, min_sec=0))
    hist = hist[hist.week.between(11, 30)]
    pool = B.pool_from_schedule(("31-42",), range(31, 43))
    spf = B.Split("final", hist, pool, users, {}, 30)
    fbf = B.popularity(spf)
    base = pd.read_csv(RES / "submission_baseline.csv", dtype=str).tv_show_id.str.split()
    man = []
    for i, r in enumerate(RULES, 1):
        p = recommend(a, range(11, 31), pool, users, r, fbf)
        B.check_preds(p, users, pool)
        fn = f"submission_l{i:02d}.csv"
        PD.write_sub(p, pool, fn)
        x = pd.read_csv(RES / fn, dtype=str).tv_show_id.str.split()
        rr = tab[tab.ranking.str.startswith(f"l0{i}")]
        own = f"T{i} {NAMES[r]}"
        man.append({"file": f"results/{fn}", "ranking": NAMES[r],
                    "main (T1 seconds)": round(rr[rr.split == "main"].iloc[0, 2], 5),
                    "secondary (T1 seconds)": round(rr[rr.split == "secondary"].iloc[0, 2], 5),
                    "main (own target)": round(rr[rr.split == "main"][own].iloc[0], 5),
                    "secondary (own target)": round(rr[rr.split == "secondary"][own].iloc[0], 5),
                    "overlap_with_baseline": round(sum(len(set(p_) & set(q)) for p_, q in zip(x, base)) / (5 * len(x)), 3)})
    man = pd.DataFrame(man)
    man.to_csv(RES / "phaseL_manifest.csv", index=False)
    identical = (RES / "submission_l01.csv").read_bytes() == (RES / "submission_baseline.csv").read_bytes()

    md = ["## 18. Phase L: probing the target definition (generated by src/phase_l.py)\n",
          "Five personal-history rankings differing only in the per-show quantity (history weeks; ties -> total "
          "seconds, then id; l01 ties by id only as in Phase A), each with the Phase A pool-popularity fallback. "
          "Target definitions T1-T5 are the same quantities over watched airings (>= 300 s) in the target window "
          "(T1 = the existing target). History rules l02-l04 count watched airings (>= 300 s); l01 and l05 use all "
          "labelled overlap.\n",
          "MAP@5, ranking rule (rows) x target definition (columns):\n"]
    for split in B.SPLITS:
        md += [f"**{split}**\n", tab[tab.split == split].drop(columns="split").to_markdown(index=False, floatfmt=".5f"), ""]
    md += [f"Check: `submission_l01.csv` byte-identical to `submission_baseline.csv`: **{identical}**.\n",
           "Files (refit on weeks 11-30, pool 6,632 from the 31-42 schedule, validated):\n",
           man.to_markdown(index=False), "", f"Phase L runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseL.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
