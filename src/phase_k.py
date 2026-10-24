"""Phase K: bagged v2 re-ranker (graded regressor, lr 0.05, 15 leaves, max_iter 200, early stopping).

Member b (b = 0..9): random_state = b (early-stopping split) and a bootstrap resample of USERS
(np.random.default_rng(b), sampled with replacement over users with a target in the training windows;
each sampled user keeps all candidate rows in every training window, duplicated per draw).
Bagged score = mean of the 10 members' predictions. Same windows and leak rule as Phase J.
Rank-average blends: score = -(sum_m a_m * rank_m), valid = union (Phase H/J construction).
Writes results/phaseK.md, results/phaseK_manifest.csv, results/submission_k01..k04.csv.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import data
import join
import phase_d as PD
import rerank as R
import scoring as SC
from cbf import item_table
from phase_j import ALLF, EVAL, NEW_COMPS, NEWF, WINDOWS, allowed, has_target

RES = data.ROOT / "results"
CFG = (0.05, 15, 200)
SEEDS = range(10)
NOSLOT = [i for i in NEWF if ALLF[i] not in ("exp_slot_pw", "rank_slot")]


def fit_one(ws, cols, seed, boot):
    Xs, ys = [], []
    users = np.asarray(ws[0]["sp"].users, dtype=np.uint64)
    if boot:
        cnt = np.bincount(np.random.default_rng(seed).integers(0, len(users), len(users)), minlength=len(users))
    for w in ws:
        m = has_target(w)
        rep = cnt[w["r"]] if boot else np.ones(len(w["r"]), dtype=int)
        rep = np.where(m, rep, 0)
        Xs.append(np.repeat(w["X"][:, cols], rep, axis=0))
        ys.append(np.repeat(w["yg"], rep))
    X, y = np.vstack(Xs), np.concatenate(ys)
    ci = [ALLF[i] for i in cols].index("category")
    X[:, ci] += 1
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[ci] = True
    mdl = HistGradientBoostingRegressor(learning_rate=CFG[0], max_leaf_nodes=CFG[1], max_iter=CFG[2],
                                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                        categorical_features=cat, random_state=seed)
    mdl.fit(X, y)
    mdl.cols, mdl.ci = cols, ci
    return mdl


def predict(mdls, w):
    X = w["X"].copy()
    out = 0
    for m in mdls:
        Xm = X[:, m.cols].copy()
        Xm[:, m.ci] += 1
        out = out + m.predict(Xm)
    return out / len(mdls)


def sv(w, sc):
    _, S, V = R.preds_from_scores(w, sc)
    return S, V


def rank_blend(w, parts):
    """parts: list of (S, V, weight). Returns top-5 dict."""
    Rsum, V = 0, None
    for S, v, a in parts:
        Rsum = Rsum + a * R.rank_rows(S, v)
        V = v.copy() if V is None else (V | v)
    return SC.topk(-Rsum, V, w["sp"], w["fb"])


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    it = item_table(("11-20", "21-30", "31-42"))
    cats = pd.Series(pd.Categorical(it.tv_show_category.fillna("?")).codes, index=it.index)
    W = {k: R.build(k, itx, hw, tw, users, cats, topn=50, cand_comps=NEW_COMPS, extra=True)
         for k, (hw, tw) in WINDOWS.items()}
    print(f"windows built {time.time() - T0:.0f}s")
    used = {s: allowed(s) for s in ("main", "secondary", "test")}
    target_w = {"main": W["W1"], "secondary": W["W2"], "test": W["TEST"]}

    res, fit_s = {}, {}
    for s in ("main", "secondary", "test"):
        ws = [W[k] for k in used[s]]
        w = target_w[s]
        t = time.time()
        single = fit_one(ws, NEWF, 0, boot=False)  # = Phase J best v2 (j02 construction)
        bag = [fit_one(ws, NEWF, b, boot=True) for b in SEEDS]
        bag_ns = [fit_one(ws, NOSLOT, b, boot=True) for b in SEEDS]
        fit_s[s] = time.time() - t
        res[s] = {"v2": sv(w, predict([single], w)), "bag": sv(w, predict(bag, w)), "bag_ns": sv(w, predict(bag_ns, w)),
                  "f03": w["f03"], "iters": [m.n_iter_ for m in bag]}
        print(s, f"fits {fit_s[s]:.0f}s")

    def methods(s):
        w, r = target_w[s], res[s]
        f = r["f03"]
        return {"j02-equivalent (single v2 + f03 0.5/0.5)": rank_blend(w, [(*r["v2"], 0.5), (*f, 0.5)]),
                "single v2 alone (j01-equivalent)": SC.topk(*r["v2"], w["sp"], w["fb"]),
                "bagged v2 alone (k04)": SC.topk(*r["bag"], w["sp"], w["fb"]),
                "bagged v2 + f03 0.5/0.5 (k01)": rank_blend(w, [(*r["bag"], 0.5), (*f, 0.5)]),
                "bagged no-slot v2 alone": SC.topk(*r["bag_ns"], w["sp"], w["fb"]),
                "bagged no-slot v2 + f03 0.5/0.5 (k02)": rank_blend(w, [(*r["bag_ns"], 0.5), (*f, 0.5)]),
                "bagged v2 0.25 + bagged no-slot 0.25 + f03 0.5 (k03)":
                    rank_blend(w, [(*r["bag"], 0.25), (*r["bag_ns"], 0.25), (*f, 0.5)])}

    comp = []
    for s in EVAL:
        w = target_w[s]
        ms = methods(s)
        ref = SC.ap_per_user(w["sp"].target, ms["j02-equivalent (single v2 + f03 0.5/0.5)"])[1]
        for name, p in ms.items():
            a = SC.ap_per_user(w["sp"].target, p)[1]
            d = a - ref
            lo, hi = SC.bootstrap_ci(d)
            comp.append({"split": s, "method": name, "MAP@5": a.mean(),
                         "Δ vs j02-eq": "—" if name.startswith("j02") else f"{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"})
    comp = pd.DataFrame(comp)
    wide = comp.pivot(index="method", columns="split", values="MAP@5")
    wide["mean"] = wide.mean(axis=1)
    for s in EVAL:
        wide[f"Δ {s} vs j02-eq"] = comp[comp.split == s].set_index("method")["Δ vs j02-eq"]
    wide = wide.sort_values("mean", ascending=False)

    wt = W["TEST"]
    pool = np.asarray(wt["sp"].pool)
    mt = methods("test")
    j02 = pd.read_csv(RES / "submission_j02.csv", dtype=str).tv_show_id.str.split()
    man = []
    for fn, key in [("submission_k01.csv", "bagged v2 + f03 0.5/0.5 (k01)"),
                    ("submission_k02.csv", "bagged no-slot v2 + f03 0.5/0.5 (k02)"),
                    ("submission_k03.csv", "bagged v2 0.25 + bagged no-slot 0.25 + f03 0.5 (k03)"),
                    ("submission_k04.csv", "bagged v2 alone (k04)")]:
        p = mt[key]
        PD.write_sub(p, pool, fn)
        x = pd.read_csv(RES / fn, dtype=str).tv_show_id.str.split()
        man.append({"file": f"results/{fn}", "method": key, "trained_on": "+".join(used["test"]),
                    "bagging": "10 members, seeds 0-9, user bootstrap", "main": round(wide.loc[key, "main"], 5),
                    "secondary": round(wide.loc[key, "secondary"], 5), "mean": round(wide.loc[key, "mean"], 5),
                    "overlap_with_j02": round(sum(len(set(a) & set(b)) for a, b in zip(x, j02)) / (5 * len(x)), 3)})
    man = pd.DataFrame(man)
    man.to_csv(RES / "phaseK_manifest.csv", index=False)
    # check: the TEST j02-equivalent rebuilt here vs the Phase J file
    pj = mt["j02-equivalent (single v2 + f03 0.5/0.5)"]
    sub = data.load_submission().user_id.tolist()
    same_j02 = np.mean([" ".join(map(str, pj[u])) == b for u, b in zip(sub, pd.read_csv(RES / "submission_j02.csv", dtype=str).tv_show_id)])

    md = ["## 17. Phase K: bagged v2 re-ranker (generated by src/phase_k.py)\n",
          f"Bagging: 10 graded HistGradientBoostingRegressor members (lr {CFG[0]}, {CFG[1]} leaves, max_iter {CFG[2]}, "
          "early stopping), member b: random_state = b and a bootstrap resample of users (rng seed b; all candidate "
          "rows of a sampled user, duplicated per draw); scores averaged. No-slot variant drops exp_slot_pw and rank_slot. "
          "Training windows: " + "; ".join(f"{s}: {'+'.join(v)}" for s, v in used.items()) + ".\n",
          f"Member iterations (main): {res['main']['iters']}; fit time for single + 2x10 members: "
          + ", ".join(f"{s} {v:.0f}s" for s, v in fit_s.items()) + ".\n",
          "Comparison (MAP@5; paired bootstrap vs j02-equivalent, 1000 resamples, seed 0):\n",
          wide.to_markdown(floatfmt=".5f"), "",
          f"Check: the j02-equivalent rebuilt on TEST matches results/submission_j02.csv on {same_j02:.1%} of rows.\n",
          "Files (validated):\n", man.to_markdown(index=False), "",
          f"Phase K runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseK.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
