"""Phase E: coarse habit components (daypart / channel exposure), reciprocal-rank fusion, and a small
diverse candidate set of submissions. Writes results/phaseE.md, results/phaseE_grid.csv,
results/phaseE_single.csv, results/phaseE_manifest.csv, results/submission_e01.csv ...

Configs: Hybrid5 base (pers 1, rate 0.25, AR 0.25, MF 1) + one habit component
{slot hl4, daypart hl None/4/8, channel hl None/4/8} with weight {0, 0.25, 0.5, 1}, fused either by
per-user min-max (as before) or by reciprocal-rank fusion sum_c w_c / (60 + rank_c). Ranked by the mean
of main and secondary MAP@5. Final submission: highest Kaggle PRIVATE score among submitted files (exam rule; test-selected).
"""
import dataclasses
import itertools
import time

import numpy as np
import pandas as pd

import baselines as B
import data
import join
import phase_d as PD
import scoring as SC
from ar import ar_scores
from exposure import exposure_scores, rate_ep_scores, rate_scores
from metrics import mapk
from mf import mf_scores

RES = data.ROOT / "results"
BASE = {"pers": 1.0, "rate": 0.25, "ar": 0.25, "mf": 1.0}
H5 = {**BASE, "slot_4": 1.0}
HABITS = ["slot_4", "daypart_None", "daypart_4", "daypart_8", "channel_None", "channel_4", "channel_8"]
W_HAB = [0.25, 0.5, 1.0]
VP = ("11-20", "21-30")
# Phase M switch: sp.hist["sec"] holds episodes; rate / AR / MF / exposure use episode units
EPISODES = False


def components(sp, hw, tw, hist_parts, target_parts):
    c = {"pers": SC.personal_scores(sp)}
    if EPISODES:
        S, v, _ = rate_ep_scores(sp, hist_parts, hw, target_parts, tw)
        c["rate"] = (S, v)
        S, v, _ = ar_scores(sp, set_min_sec=0.5)  # transaction = shows with >= 0.5 episodes
        c["ar"] = (S, v)
        # MF confidence 1 + alpha*log1p(sec/60): pass episodes*60 so that it becomes log1p(episodes)
        S, v, _ = mf_scores(dataclasses.replace(sp, hist=sp.hist.assign(sec=sp.hist.sec * 60.0)), *PD.MF_CFG)
        c["mf"] = (S, v)
    else:
        S, v, _ = rate_scores(sp, hist_parts, hw, target_parts, tw)
        c["rate"] = (S, v)
        S, v, _ = ar_scores(sp)
        c["ar"] = (S, v)
        S, v, _ = mf_scores(sp, *PD.MF_CFG)
        c["mf"] = (S, v)
    parts = tuple(dict.fromkeys(hist_parts + target_parts))
    rt = {}
    for h in HABITS:
        gran, hl = h.split("_")
        t = time.time()
        S, v, _ = exposure_scores(sp, hw, parts, tw, None if hl == "None" else int(hl), gran, per_airing=EPISODES)
        c[h] = (S, v)
        rt[h] = time.time() - t
    return c, rt


def fuse(comps, w, how="minmax"):
    S, V = None, None
    for name, wt in w.items():
        if wt == 0:
            continue
        s, v = comps[name]
        z = wt * (SC.minmax_rows(s, v) if how == "minmax" else SC.rrf_rows(s, v))
        S = z if S is None else S + z
        V = v.copy() if V is None else (V | v)
    return S, V


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    sp, comps, fb, rts = {}, {}, {}, {}
    for s, (hw, tw) in B.SPLITS.items():
        sp[s] = B.make_split(s, itx, 0, users)
        fb[s] = B.popularity(sp[s])
        comps[s], rts[s] = components(sp[s], list(hw), list(tw), VP, VP)

    def ap(s, w, how):
        return SC.ap_per_user(sp[s].target, SC.topk(*fuse(comps[s], w, how), sp[s], fb[s]))[1]

    # single habit components
    single = []
    for h in HABITS:
        r = {"component": h}
        for s in sp:
            r[s] = ap(s, {h: 1.0}, "minmax").mean()
        r["mean"] = (r["main"] + r["secondary"]) / 2
        r["fit_s_main"] = rts["main"][h]
        single.append(r)
    single = pd.DataFrame(single).sort_values("mean", ascending=False)
    single.to_csv(RES / "phaseE_single.csv", index=False)

    # hybrid grid
    ap_h5 = {s: ap(s, H5, "minmax") for s in sp}
    rows = []
    for how in ("minmax", "rank"):
        cfgs = [("none", 0.0)] + list(itertools.product(HABITS, W_HAB))
        for h, wt in cfgs:
            w = dict(BASE, **({h: wt} if wt else {}))
            r = {"habit": h, "w_habit": wt, "fusion": how}
            for s in sp:
                a = ap(s, w, how)
                d = a - ap_h5[s]
                lo, hi = SC.bootstrap_ci(d)
                r[s] = a.mean()
                r[f"d_{s}"] = f"{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"
            r["mean"] = (r["main"] + r["secondary"]) / 2
            rows.append(r)
    g = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    g.to_csv(RES / "phaseE_grid.csv", index=False)

    # candidate selection (at most 8)
    key = lambda r: (r["habit"], float(r["w_habit"]), r["fusion"])
    picks, why = [], {}

    def add(r, reason):
        k = key(r)
        if k in why:
            why[k] += "; " + reason
            return
        if len(picks) < 8:
            picks.append(r)
            why[k] = reason

    mm = g[g.fusion == "minmax"]
    for wt in (0.0, 0.25, 0.5):
        r = mm[(mm.habit == ("none" if wt == 0 else "slot_4")) & (mm.w_habit == wt)].iloc[0]
        add(r, f"slot ladder w={wt}")
    add(g[g.habit.str.startswith("daypart")].iloc[0], "best daypart by mean")
    add(g[g.habit.str.startswith("channel")].iloc[0], "best channel by mean")
    rk = g[g.fusion == "rank"]
    best_rank = next((r for _, r in rk.iterrows() if key(r) not in why), rk.iloc[0])
    add(rk.iloc[0] if key(rk.iloc[0]) not in why else best_rank, "best rank fusion by mean")
    add(g.iloc[0], "overall best by mean")

    # refit on weeks 11-30 with the 31-42 schedule
    t = time.time()
    hist = join.user_show_week(join.watched(itx, min_sec=0))
    hist = hist[hist.week.between(11, 30)]
    pool = B.pool_from_schedule(("31-42",), range(31, 43))
    spf = B.Split("final", hist, pool, users, {}, 30)
    cf, _ = components(spf, list(range(11, 31)), list(range(31, 43)), VP, ("31-42",))
    fbf = B.popularity(spf)
    man = []
    for i, r in enumerate(picks, 1):
        w = dict(BASE, **({r["habit"]: float(r["w_habit"])} if r["w_habit"] else {}))
        p = SC.topk(*fuse(cf, w, r["fusion"]), spf, fbf)
        B.check_preds(p, users, pool)
        fn = f"submission_e{i:02d}.csv"
        PD.write_sub(p, pool, fn)
        gran, _, hl = r["habit"].partition("_")
        man.append({"file": f"results/{fn}", "components": "+".join(k for k, v in w.items() if v),
                    "weights": " ".join(f"{k}={v:g}" for k, v in w.items() if v), "fusion": r["fusion"],
                    "half_life": hl if r["habit"] != "none" else "", "main": round(r["main"], 5),
                    "secondary": round(r["secondary"], 5), "mean": round(r["mean"], 5),
                    "why": why[key(r)]})
    man = pd.DataFrame(man)
    man.to_csv(RES / "phaseE_manifest.csv", index=False)

    h5m, h5s = ap_h5["main"].mean(), ap_h5["secondary"].mean()
    md = ["## 11. Phase E: coarse habit components, rank fusion, candidate set (generated by src/phase_e.py)\n",
          "- **Channel exposure**: slot = channel. **Daypart exposure**: slot = (channel, weekday/weekend, "
          "4-hour block) = 12 slots per channel. Same construction as the slot model (profile from ALL history "
          "views clipped at the target-window start, / (Σ week weights × 3600); item airtime from the "
          "target-window schedule; score = expected seconds).",
          "- **Rank fusion (RRF)**: score = Σ_c w_c / (60 + rank_c(item)) over each component's valid items "
          "(rank 1 = highest score, ties → smaller id); valid = union of components with w > 0; fallback fill as always.",
          f"- Base = Hybrid5 without exposure: {BASE}. Reference Hybrid5 (min-max, slot_4 = 1): main {h5m:.5f}, "
          f"secondary {h5s:.5f}, mean {(h5m + h5s) / 2:.5f}.",
          "- Ranking: mean of main and secondary MAP@5; Δ = paired bootstrap vs Hybrid5 (1000 resamples, seed 0).\n",
          "Single habit components (alone, min-max = plain ranking):\n",
          single.to_markdown(index=False, floatfmt=".5f"), "",
          f"Hybrid grid ({len(g)} configs), top 15:\n",
          g.head(15).to_markdown(index=False, floatfmt=".5f"), "",
          "Min-max slot ladder and best per habit family:\n",
          pd.concat([mm[mm.habit.isin(["none", "slot_4"])].sort_values("w_habit"),
                     g.groupby(g.habit.str.split("_").str[0] + "_" + g.fusion).head(1)])
          .drop_duplicates().to_markdown(index=False, floatfmt=".5f"), "",
          f"Candidate submissions (refit weeks 11-30, pool {len(pool):,} from the 31-42 schedule, all validated; "
          f"refit {time.time() - t:.1f}s):\n",
          man.to_markdown(index=False), "",
          "**Selection rule:** Methods are compared on the temporal validation; among submitted files the designated final submission is the one with the highest Kaggle PRIVATE score, as the exam rules specify; this makes the reported private score optimistic (test-selected).\n",
          f"Phase E runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
