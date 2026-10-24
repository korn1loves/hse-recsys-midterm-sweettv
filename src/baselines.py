"""Phase A steps 4-6: validation splits, targets, candidate pool and reference recommenders.
Writes results/baselines.md and results/baselines.csv.

Protocol shared by every method (also later phases):
- history = watched labelled airings in history weeks; target = top-5 labelled shows by watched
  seconds in target weeks (hypothesis for the competition's 'preference');
- candidate pool = labelled shows with an airing that STARTS in a target week (schedule only);
- repeats allowed (shows already watched in history can be recommended);
- fallback = pool popularity by watched seconds over all history (`pop_pool`), used to fill
  every list to 5 items;
- eval users = submission users with a non-empty target; metric MAP@5 (src/metrics.py).
"""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import data
import join
from metrics import mapk

K = 5
TARGET_MIN_SEC = 300          # threshold that defines the target (fixed across methods)
SPLITS = {"main": (range(11, 21), range(21, 31)), "secondary": (range(11, 25), range(25, 31))}


@dataclass
class Split:
    name: str
    hist: pd.DataFrame        # user_id, tv_show_id, week, sec (watched, history weeks)
    pool: np.ndarray          # candidate tv_show_ids
    users: list               # users to predict for
    target: dict              # user -> list of target show ids (eval users only)
    ref_week: int             # last history week (recency reference)


def top_by(df, score, key="tv_show_id"):
    return df.sort_values([score, key], ascending=[False, True])[key].tolist()


def pool_from_schedule(parts, weeks) -> np.ndarray:
    s = data.load_schedule_all(parts)
    s = s[(s.tv_show_id != 0) & s.week.isin(list(weeks))]
    return np.sort(s.tv_show_id.unique())


def make_target(itx, weeks, users, min_sec=TARGET_MIN_SEC) -> dict:
    w = join.watched(itx, min_sec=min_sec)
    w = w[w.week.isin(list(weeks)) & w.user_id.isin(users)]
    g = w.groupby(["user_id", "tv_show_id"], as_index=False).overlap_s.sum()
    g = g.sort_values(["user_id", "overlap_s", "tv_show_id"], ascending=[True, False, True])
    return {u: grp.tv_show_id.head(K).tolist() for u, grp in g.groupby("user_id")}


def make_split(name, itx, hist_min_sec, users) -> Split:
    hw, tw = SPLITS[name]
    hist = join.user_show_week(join.watched(itx, min_sec=hist_min_sec))
    hist = hist[hist.week.isin(list(hw))]
    pool = pool_from_schedule(("11-20", "21-30"), tw)
    target = make_target(itx, tw, users)
    return Split(name, hist, pool, users, target, max(hw))


# ---------------------------------------------------------------- recommenders
def recency_w(week, ref_week, half_life):
    if half_life is None:
        return np.ones(len(week))
    return 0.5 ** ((ref_week - np.asarray(week)) / half_life)


def popularity(sp: Split, in_pool=True, half_life=None, by="sec"):
    h = sp.hist
    if in_pool:
        h = h[h.tv_show_id.isin(sp.pool)]
    w = recency_w(h.week, sp.ref_week, half_life)
    val = h.sec.values if by == "sec" else np.ones(len(h))
    if by == "users":  # distinct users (recency weight of the user's latest week on the show)
        h = h.assign(w=w).groupby(["tv_show_id", "user_id"]).w.max().reset_index()
        sc = h.groupby("tv_show_id").w.sum()
    else:
        sc = pd.Series(val * w, index=h.tv_show_id.values).groupby(level=0).sum()
    return top_by(sc.rename("s").reset_index().rename(columns={"index": "tv_show_id"}), "s")


def fill(lst, fallback, k=K):
    out = list(dict.fromkeys(lst))[:k]
    for x in fallback:
        if len(out) >= k:
            break
        if x not in out:
            out.append(x)
    return out


def rec_global(sp, items):
    return {u: fill(items, items) for u in sp.users}


def rec_personal(sp, fallback, half_life=None):
    h = sp.hist[sp.hist.tv_show_id.isin(sp.pool) & sp.hist.user_id.isin(sp.users)]
    h = h.assign(s=h.sec * recency_w(h.week, sp.ref_week, half_life))
    g = h.groupby(["user_id", "tv_show_id"], as_index=False).s.sum()
    g = g.sort_values(["user_id", "s", "tv_show_id"], ascending=[True, False, True])
    per = {u: grp.tv_show_id.head(K).tolist() for u, grp in g.groupby("user_id")}
    return {u: fill(per.get(u, []), fallback) for u in sp.users}


def methods(sp: Split):
    """name -> callable producing {user: [5 ids]}; fallback is pop_pool for everything."""
    fb = popularity(sp, in_pool=True)
    m = {
        "a_pop_all": lambda: rec_global(sp, fill(popularity(sp, in_pool=False), fb)),
        "b_pop_pool": lambda: rec_global(sp, fb),
        "b2_pop_pool_users": lambda: rec_global(sp, fill(popularity(sp, by="users"), fb)),
    }
    for hl in (1, 2, 4):
        m[f"c_pop_pool_hl{hl}"] = (lambda hl=hl: rec_global(sp, fill(popularity(sp, half_life=hl), fb)))
    m["d_personal"] = lambda: rec_personal(sp, fb)
    for hl in (1, 2, 4):
        m[f"d_personal_hl{hl}"] = (lambda hl=hl: rec_personal(sp, fb, half_life=hl))
    return m


def check_preds(preds, users, pool=None):
    pool = set(pool) if pool is not None else None
    for u in users:
        p = preds[u]
        assert len(p) == K and len(set(p)) == K and all(x != 0 for x in p), u
        assert pool is None or all(x in pool for x in p), u


def main():
    t0 = time.time()
    itx = join.airing_interactions()
    sub_users = data.load_submission().user_id.tolist()
    rows, lines = [], ["# Reference baselines (generated by src/baselines.py)\n"]
    for split in SPLITS:
        hw, tw = SPLITS[split]
        sp0 = make_split(split, itx, 300, sub_users)
        tgt_items = {x for v in sp0.target.values() for x in v}
        hist_users = set(sp0.hist.user_id)
        n_has_hist = sum(1 for u in sp0.target if u in hist_users)
        seen = sp0.hist.groupby("user_id").tv_show_id.agg(set).to_dict()
        lens = pd.Series([len(v) for v in sp0.target.values()])
        lines += [f"## Split `{split}`: history weeks {min(hw)}-{max(hw)} -> target {min(tw)}-{max(tw)}\n",
                  f"- eval users (submission users with non-empty target @ {TARGET_MIN_SEC}s): "
                  f"{len(sp0.target):,}/{len(sub_users):,}; with watched history (>=300s): {n_has_hist:,}",
                  f"- target length distribution: {lens.value_counts().sort_index().to_dict()}",
                  f"- candidate pool size: {len(sp0.pool):,}; target items outside pool: "
                  f"{len(tgt_items - set(sp0.pool))}/{len(tgt_items)}",
                  f"- share of eval users' target items already watched by the user in history: "
                  f"{np.mean([x in seen.get(u, ()) for u, v in sp0.target.items() for x in v]):.2%}",
                  ""]
        for thr in (0, 300, 900):
            sp = make_split(split, itx, thr, sub_users)
            for name, fn in methods(sp).items():
                t1 = time.time()
                preds = fn()
                rt = time.time() - t1
                check_preds(preds, sp.users, None if name == "a_pop_all" else sp.pool)
                rows.append({"split": split, "hist_min_sec": thr, "method": name,
                             "MAP@5": mapk(sp.target, preds), "runtime_s": rt})
                print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(data.ROOT / "results" / "baselines.csv", index=False)
    lines.append("## MAP@5 (target threshold fixed at 300 s; columns = history watch threshold)\n")
    for split in SPLITS:
        t = res[res.split == split].pivot(index="method", columns="hist_min_sec", values="MAP@5")
        t.columns = [f"hist>={c}s" for c in t.columns]
        lines += [f"### {split}\n", t.to_markdown(floatfmt=".5f"), ""]
    rt = res.groupby("method").runtime_s.mean()
    lines += ["Mean runtime per fit+predict (s):\n", rt.to_frame().to_markdown(floatfmt=".2f"), ""]

    # target-definition sensitivity for the best-by-main methods
    best = res[(res.split == "main")].sort_values("MAP@5", ascending=False).iloc[0]
    lines.append(f"## Target-threshold sensitivity (history threshold {best.hist_min_sec}s)\n")
    srows = []
    for split in SPLITS:
        sp = make_split(split, itx, int(best.hist_min_sec), sub_users)
        ms = methods(sp)
        for tthr in (0, 300, 900):
            tw = SPLITS[split][1]
            tgt = make_target(itx, tw, sub_users, min_sec=tthr)
            for name in ("b_pop_pool", "c_pop_pool_hl1", "d_personal", best.method):
                srows.append({"split": split, "target_min_sec": tthr, "method": name,
                              "n_users": len(tgt), "MAP@5": mapk(tgt, ms[name]())})
    s = pd.DataFrame(srows).drop_duplicates(["split", "target_min_sec", "method"])
    lines += [s.pivot(index=["split", "method"], columns="target_min_sec", values="MAP@5")
              .to_markdown(floatfmt=".5f"), "",
              f"eval users per target threshold: "
              f"{s.groupby(['split', 'target_min_sec']).n_users.first().to_dict()}", ""]
    lines.append(f"Total runtime {time.time() - t0:.1f}s")
    (data.ROOT / "results" / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
