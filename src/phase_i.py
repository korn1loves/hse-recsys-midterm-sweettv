"""Phase I: rank-level ensembles of diverse members.

Members (score matrices over the window pool; items outside a member's valid set are unranked):
f03, e05, f06 (min-max hybrids) and h01, h04, h05 (re-rankers; leak-free: main trained on W0,
secondary on W0+W1b, TEST on W0+W1+W2).
Ensemble scores per user over each member's FULL ranking (rank 1 = best, ties -> smaller id):
- RRF-style: sum_m w_m / (k + rank_m), k in {5, 20}; unranked items contribute 0;
- Borda top-30: sum_m w_m * (31 - rank_m) for rank_m <= 30, else 0.
Valid = union of members' valid items; top-5 + fallback as always.
Reference "h02-equivalent" = Phase H blend: -(0.5 rank_h01 + 0.5 rank_f03) (exactly how h02 was built).
Writes results/phaseI.md, results/phaseI_grid.csv, results/phaseI_manifest.csv, results/submission_i0*.csv.
"""
import time

import numpy as np
import pandas as pd

import data
import join
import phase_d as PD
import phase_e as PE
import rerank as R
import scoring as SC
from cbf import item_table
from phase_e import BASE
from phase_g import F03
from phase_h import WINDOWS
from phase_h2 import KEEP, fit, predict

RES = data.ROOT / "results"
PE.HABITS = ["slot_4", "daypart_4", "channel_4", "daypart_None"]
FUS = {"e05": {**BASE, "channel_4": 0.5}, "f06": {**BASE, "daypart_None": 1.0}}
H01, H04, H05 = (0.05, 15, 200), (0.05, 15, 200), (0.05, 31, 200)
TRAIN = {"main": (["W0"], "W1"), "secondary": (["W0", "W1b"], "W2"), "test": (["W0", "W1", "W2"], "TEST")}


def ens_scores(members, w, k):
    S = None
    V = None
    for name, wt in w.items():
        s, v = members[name]
        rk = R.rank_rows(s, v)
        if k == "borda":
            z = np.where(v & (rk <= 30), 31 - rk, 0).astype(float)
        else:
            z = np.where(v, 1.0 / (k + rk), 0.0)
        S = wt * z if S is None else S + wt * z
        V = v.copy() if V is None else (V | v)
    return S, V


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    it = item_table(("11-20", "21-30", "31-42"))
    cats = pd.Series(pd.Categorical(it.tv_show_category.fillna("?")).codes, index=it.index)
    WA, WB = {}, {}
    for k, (hw, tw) in WINDOWS.items():
        R.CAND_COMPS = ["pers", "rate", "daypart_4", "channel_4", "slot_4", "ar", "mf"]
        WA[k] = R.build(k, itx, hw, tw, users, cats, FUS)
        R.CAND_COMPS = ["pers", "rate", "daypart_4", "channel_4", "ar", "mf"]
        WB[k] = R.build(k, itx, hw, tw, users, cats)
        print(k, f"{time.time() - T0:.0f}s")
    allcols = list(range(len(R.FEATURES)))

    M, W = {}, {}
    for s, (tr, te) in TRAIN.items():
        w = WA[te]
        W[s] = w
        _, S1, V1 = predict(fit([WA[k] for k in tr], H01, allcols), w)
        _, S4, V4 = predict(fit([WA[k] for k in tr], H04, KEEP), w)
        _, S5, V5 = predict(fit([WB[k] for k in tr], H05, KEEP), WB[te])
        M[s] = {"f03": w["f03"], "e05": w["fus"]["e05"], "f06": w["fus"]["f06"],
                "h01": (S1, V1), "h04": (S4, V4), "h05": (S5, V5)}

    def preds(s, w, k):
        return SC.topk(*ens_scores(M[s], w, k), W[s]["sp"], W[s]["fb"])

    def h02eq(s):
        S1, V1 = M[s]["h01"]
        return R.blend(W[s], S1, V1)

    configs = []
    for a in (0.3, 0.5, 0.7):
        for k in (5, 20, "borda"):
            configs.append((f"(a) h01 {a:g} + f03 {1 - a:g}", {"h01": a, "f03": round(1 - a, 2)}, k))
    for k in (5, 20, "borda"):
        configs.append(("(b) h01 + h05 + f03", {"h01": 1, "h05": 1, "f03": 1}, k))
        configs.append(("(c) h01 + h05 + f03 + e05", {"h01": 1, "h05": 1, "f03": 1, "e05": 1}, k))
        configs.append(("(d) h01 + h04 + h05 + f03 + f06", {"h01": 1, "h04": 1, "h05": 1, "f03": 1, "f06": 1}, k))

    ref = {s: SC.ap_per_user(W[s]["sp"].target, h02eq(s))[1] for s in ("main", "secondary")}
    single = {s: {m: SC.ap_per_user(W[s]["sp"].target, SC.topk(*M[s][m], W[s]["sp"], W[s]["fb"]))[1].mean()
                  for m in M[s]} for s in ("main", "secondary")}
    rows = []
    for name, w, k in configs:
        r = {"ensemble": name, "k": str(k), "weights": " ".join(f"{m}={v:g}" for m, v in w.items()), "_w": w, "_k": k}
        for s in ("main", "secondary"):
            a = SC.ap_per_user(W[s]["sp"].target, preds(s, w, k))[1]
            d = a - ref[s]
            lo, hi = SC.bootstrap_ci(d)
            r[s] = a.mean()
            r[f"Δ {s} vs h02-eq"] = f"{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"
        r["mean"] = (r["main"] + r["secondary"]) / 2
        rows.append(r)
    g = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    g.drop(columns=["_w", "_k"]).to_csv(RES / "phaseI_grid.csv", index=False)

    best_of = lambda prefix: g[g.ensemble.str.startswith(prefix)].iloc[0]
    two = g[g.ensemble.str.startswith("(a)") & ~g.ensemble.str.contains("0.5")].iloc[0]
    picks = [(best_of("(b)"), "(b) three-member, best k by mean"), (best_of("(c)"), "(c) four-member, best k by mean"),
             (best_of("(d)"), "(d) five-member, best k by mean"), (two, "best two-member weight variant other than 0.5/0.5"),
             (g.iloc[0], "(e) overall best by mean")]

    t = time.time()
    pool = np.asarray(W["test"]["sp"].pool)
    h02 = pd.read_csv(RES / "submission_h02.csv", dtype=str).tv_show_id.str.split()
    man, seen = [], {}
    for r, why in picks:
        key = (r.ensemble, r.k)
        if key in seen:
            man[seen[key]]["why"] += "; " + why
            continue
        p = preds("test", r._w, r._k)
        fn = f"submission_i{len(man) + 1:02d}.csv"
        PD.write_sub(p, pool, fn)
        x = pd.read_csv(RES / fn, dtype=str).tv_show_id.str.split()
        ov = sum(len(set(a) & set(b)) for a, b in zip(x, h02)) / (5 * len(x))
        seen[key] = len(man)
        man.append({"file": f"results/{fn}", "members": "+".join(r._w), "weights": r.weights, "k": r.k,
                    "main": round(r.main, 5), "secondary": round(r.secondary, 5), "mean": round(r["mean"], 5),
                    "overlap_with_h02": round(ov, 3), "why": why})
    man = pd.DataFrame(man)
    man.to_csv(RES / "phaseI_manifest.csv", index=False)

    md = ["## 15. Phase I: rank-level ensembles (generated by src/phase_i.py)\n",
          "Members: f03, e05 (base + channel_4 0.5), f06 (base + daypart_None 1), h01 (re-ranker), h04 (no-slot "
          "re-ranker), h05 (no-slot re-ranker, no slot candidates); re-rankers leak-free (main: W0; secondary: "
          "W0+W1b; TEST: W0+W1+W2). Ensemble = Σ w/(k + rank) over each member's full ranking (k ∈ {5, 20}) or "
          "Borda on the top 30; valid = union of members; top-5 + fallback.\n",
          f"Reference h02-equivalent (Phase H blend, −(0.5 rank_h01 + 0.5 rank_f03)): main {ref['main'].mean():.5f}, "
          f"secondary {ref['secondary'].mean():.5f}.\n",
          "Single members (MAP@5):\n",
          pd.DataFrame(single).round(5).to_markdown(), "",
          "Ensembles (Δ = paired bootstrap vs h02-equivalent, 1000 resamples, seed 0):\n",
          g.drop(columns=["_w", "_k"]).to_markdown(index=False, floatfmt=".5f"), "",
          f"Files (refit on TEST, validated; {time.time() - t:.1f}s):\n", man.to_markdown(index=False), "",
          f"Phase I runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseI.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
