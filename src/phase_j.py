"""Phase J: re-ranker v2 (more windows, top-50 candidates + daypart_None, extra features, graded labels).

Leak rule (enforced in code): a model evaluated on a split may only be trained on windows whose target
ends before that split's target starts. Final model: all windows with targets inside weeks 16-30.
Ablation rows use the fixed Phase H setting (lr 0.05, 15 leaves, max_iter 200) to isolate ingredients.
Writes results/phaseJ.md, results/phaseJ_grid.csv, results/phaseJ_manifest.csv, results/submission_j0*.csv.
"""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import data
import join
import phase_d as PD
import phase_e as PE
import rerank as R
import scoring as SC
from cbf import item_table

RES = data.ROOT / "results"
PE.HABITS = ["slot_4", "daypart_4", "channel_4", "daypart_None"]
NEW_COMPS = R.OLD_COMPS + ["daypart_None"]
WINDOWS = {"W0": (range(11, 16), range(16, 21)), "Ws1": (range(11, 17), range(17, 23)),
           "Ws2": (range(11, 19), range(19, 25)), "W12": (range(11, 19), range(19, 31)),
           "W1b": (range(11, 21), range(21, 25)), "W1": (range(11, 21), range(21, 31)),
           "W2": (range(11, 25), range(25, 31)), "TEST": (range(11, 31), range(31, 43))}
EVAL = {"main": "W1", "secondary": "W2"}
OLD_WINDOWS = {"main": ["W0"], "secondary": ["W0", "W1b"], "test": ["W0", "W1", "W2"]}
GRID = list(itertools.product([0.05, 0.1], [15, 31], [200, 400]))
H01 = (0.05, 15, 200)
ALLF = R.FEATURES + R.EXTRA
OLDF = list(range(len(R.FEATURES)))
NEWF = list(range(len(ALLF)))


def allowed(split):
    if split == "test":
        return [k for k, (h, t) in WINDOWS.items() if k != "TEST" and min(t) >= 16 and max(t) <= 30]
    start = min(WINDOWS[EVAL[split]][1])
    return [k for k, (h, t) in WINDOWS.items() if k not in ("TEST",) and max(t) < start]


def sub(w, old_only):
    if not old_only:
        return w
    m = w["old_cand"]
    out = dict(w)
    for k in ("r", "c", "X", "y", "yg", "old_cand"):
        out[k] = w[k][m]
    return out


def fit(ws, cfg, cols, graded):
    X = np.vstack([w["X"][has_target(w)][:, cols] for w in ws])
    y = np.concatenate([(w["yg"] if graded else w["y"])[has_target(w)] for w in ws])
    ci = [ALLF[i] for i in cols].index("category")
    X[:, ci] += 1
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[ci] = True
    kw = dict(learning_rate=cfg[0], max_leaf_nodes=cfg[1], max_iter=cfg[2], early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=20, categorical_features=cat, random_state=0)
    m = HistGradientBoostingRegressor(**kw) if graded else HistGradientBoostingClassifier(**kw)
    t = time.time()
    m.fit(X, y)
    m.fit_s, m.cols, m.ci, m.graded = time.time() - t, cols, ci, graded
    return m


def has_target(w):
    tu = set(w["sp"].target)
    return np.array([u in tu for u in w["sp"].users])[w["r"]]


def scores(m, w):
    X = w["X"][:, m.cols].copy()
    X[:, m.ci] += 1
    return m.predict(X) if m.graded else m.predict_proba(X)[:, 1]


def run(m, w, blend_a=None):
    p, S, V = R.preds_from_scores(w, scores(m, w))
    if blend_a is None:
        return p, S, V
    return R.blend(w, S, V, a=blend_a), S, V


def apv(w, p):
    return SC.ap_per_user(w["sp"].target, p)[1]


def boot(a, b):
    d = a - b
    lo, hi = SC.bootstrap_ci(d)
    return f"{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]"


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    it = item_table(("11-20", "21-30", "31-42"))
    cats = pd.Series(pd.Categorical(it.tv_show_category.fillna("?")).codes, index=it.index)
    W, rec = {}, []
    for k, (hw, tw) in WINDOWS.items():
        t = time.time()
        W[k] = R.build(k, itx, hw, tw, users, cats, topn=50, cand_comps=NEW_COMPS, extra=True)
        n_t = sum(len(v) for v in W[k]["sp"].target.values())
        rec.append({"window": k, "history": f"{min(hw)}-{max(hw)}", "target": f"{min(tw)}-{max(tw)}",
                    "avg cands v2": len(W[k]["r"]) / len(users),
                    "recall v2": W[k]["y"].sum() / n_t if n_t else np.nan,
                    "avg cands old": W[k]["old_cand"].sum() / len(users),
                    "recall old": W[k]["y"][W[k]["old_cand"]].sum() / n_t if n_t else np.nan,
                    "build_s": time.time() - t})
        print(rec[-1])
    rec = pd.DataFrame(rec)
    used = {s: allowed(s) for s in ("main", "secondary", "test")}
    print("windows used:", used)

    def model(split, cfg, graded, feats=NEWF, old_cands=False, windows=None):
        ws = [sub(W[k], old_cands) for k in (windows or used[split])]
        return fit(ws, cfg, feats, graded)

    def evaluate(split, m, old_cands=False, blend_a=None):
        w = sub(W[EVAL[split]] if split != "test" else W["TEST"], old_cands)
        return run(m, w, blend_a)

    # h02-equivalent reference: Phase H setting, blended 0.5/0.5 with f03
    ref, ref_alone = {}, {}
    for s in EVAL:
        m = model(s, H01, False, OLDF, True, OLD_WINDOWS[s])
        w = sub(W[EVAL[s]], True)
        ref[s] = apv(w, run(m, w, 0.5)[0])
        ref_alone[s] = apv(w, run(m, w)[0])

    # v2 grid: both label types, alone (ranked by mean)
    rows = []
    for graded, cfg in itertools.product([False, True], GRID):
        r = {"label": "graded (regressor)" if graded else "binary (classifier)", "learning_rate": cfg[0],
             "max_leaf_nodes": cfg[1], "max_iter": cfg[2], "_g": graded, "_cfg": cfg}
        for s in EVAL:
            m = model(s, cfg, graded)
            r[s] = apv(W[EVAL[s]], evaluate(s, m)[0]).mean()
        r["mean"] = (r["main"] + r["secondary"]) / 2
        rows.append(r)
        print({k: v for k, v in r.items() if not k.startswith("_")})
    g = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    g.drop(columns=["_g", "_cfg"]).to_csv(RES / "phaseJ_grid.csv", index=False)
    best = g.iloc[0]
    bestG = g[g._g].iloc[0]

    # comparison table
    comp = []
    for name, rr in [("v2 best", best), ("v2 best graded", bestG)]:
        for s in EVAL:
            m = model(s, rr._cfg, rr._g)
            w = W[EVAL[s]]
            a = apv(w, evaluate(s, m)[0])
            b = apv(w, evaluate(s, m, blend_a=0.5)[0])
            b4 = apv(w, evaluate(s, m, blend_a=0.4)[0])
            comp += [{"split": s, "method": f"{name} ({rr.label}, lr {rr._cfg[0]}, leaves {rr._cfg[1]}) alone",
                      "MAP@5": a.mean(), "Δ vs h02-eq": boot(a, ref[s])},
                     {"split": s, "method": f"{name} + f03 rank-avg 0.5/0.5", "MAP@5": b.mean(), "Δ vs h02-eq": boot(b, ref[s])},
                     {"split": s, "method": f"{name} + f03 rank-avg 0.4/0.6", "MAP@5": b4.mean(), "Δ vs h02-eq": boot(b4, ref[s])}]
    for s in EVAL:
        comp += [{"split": s, "method": "h02-equivalent (Phase H h01 + f03 0.5/0.5)", "MAP@5": ref[s].mean(), "Δ vs h02-eq": "—"},
                 {"split": s, "method": "h01-equivalent alone", "MAP@5": ref_alone[s].mean(), "Δ vs h02-eq": boot(ref_alone[s], ref[s])}]
    comp = pd.DataFrame(comp).drop_duplicates(["split", "method"]).sort_values(["split", "MAP@5"], ascending=[True, False])

    # ablation (fixed Phase H hyper-parameters; binary unless graded is the ingredient)
    abl_cfg = [("Phase H setting (old windows, top-30 cands, old features, binary)", dict(old_cands=True, feats=OLDF, graded=False, win="old")),
               ("+ windows only", dict(old_cands=True, feats=OLDF, graded=False, win="new")),
               ("+ candidates only (top-50 + daypart_None)", dict(old_cands=False, feats=OLDF, graded=False, win="old")),
               ("+ features only", dict(old_cands=True, feats=NEWF, graded=False, win="old")),
               ("+ graded label only", dict(old_cands=True, feats=OLDF, graded=True, win="old")),
               ("full v2, binary", dict(old_cands=False, feats=NEWF, graded=False, win="new")),
               ("full v2, graded", dict(old_cands=False, feats=NEWF, graded=True, win="new"))]
    abl = []
    for name, a in abl_cfg:
        r = {"configuration": name}
        for s in EVAL:
            wins = OLD_WINDOWS[s] if a["win"] == "old" else used[s]
            m = model(s, H01, a["graded"], a["feats"], a["old_cands"], wins)
            w = sub(W[EVAL[s]], a["old_cands"])
            r[f"{s} alone"] = apv(w, run(m, w)[0]).mean()
            r[f"{s} + f03 0.5/0.5"] = apv(w, run(m, w, 0.5)[0]).mean()
            r[f"{s} windows"] = "+".join(wins)
        abl.append(r)
    abl = pd.DataFrame(abl)

    # final files
    t = time.time()
    wt = W["TEST"]
    pool = np.asarray(wt["sp"].pool)
    mb = model("test", best._cfg, best._g)
    p1, S1, V1 = run(mb, wt)
    p2 = R.blend(wt, S1, V1, a=0.5)
    p4 = R.blend(wt, S1, V1, a=0.4)
    files = [("submission_j01.csv", p1, f"v2 best alone ({best.label})", best), ("submission_j02.csv", p2, "v2 best + f03 0.5/0.5", best)]
    note = ""
    if bestG._g != best._g or bestG._cfg != best._cfg:
        mg = model("test", bestG._cfg, True)
        _, Sg, Vg = run(mg, wt)
        files.append(("submission_j03.csv", R.blend(wt, Sg, Vg, a=0.5), "v2 best graded + f03 0.5/0.5", bestG))
    else:
        note = "- j03 not written: the best v2 model already uses the graded label, so it would equal j02."
    files.append(("submission_j04.csv", p4, "v2 best 0.4 + f03 0.6", best))
    h02 = pd.read_csv(RES / "submission_h02.csv", dtype=str).tv_show_id.str.split()
    val = lambda meth, s: comp[(comp.split == s) & (comp.method == meth)]["MAP@5"].iloc[0]
    vname = {"submission_j01.csv": lambda rr: f"v2 best ({rr.label}, lr {rr._cfg[0]}, leaves {rr._cfg[1]}) alone",
             "submission_j02.csv": lambda rr: "v2 best + f03 rank-avg 0.5/0.5",
             "submission_j03.csv": lambda rr: "v2 best graded + f03 rank-avg 0.5/0.5",
             "submission_j04.csv": lambda rr: "v2 best + f03 rank-avg 0.4/0.6"}
    man = []
    for fn, p, desc, rr in files:
        PD.write_sub(p, pool, fn)
        x = pd.read_csv(RES / fn, dtype=str).tv_show_id.str.split()
        meth = vname[fn](rr)
        man.append({"file": f"results/{fn}", "method": desc, "label": rr.label,
                    "params": f"lr={rr._cfg[0]} leaves={rr._cfg[1]} max_iter={rr._cfg[2]} (early stopping)",
                    "trained_on": "+".join(used["test"]), "main": round(val(meth, "main"), 5),
                    "secondary": round(val(meth, "secondary"), 5),
                    "overlap_with_h02": round(sum(len(set(a) & set(b)) for a, b in zip(x, h02)) / (5 * len(x)), 3)})
    man = pd.DataFrame(man)
    man.to_csv(RES / "phaseJ_manifest.csv", index=False)

    md = ["## 16. Phase J: re-ranker v2 (generated by src/phase_j.py)\n",
          "Leak rule (enforced programmatically): training windows must have target end < evaluated split's target "
          "start; final = all windows with targets inside 16-30.\n",
          "Windows used: " + "; ".join(f"**{s}**: {', '.join(v)}" for s, v in used.items()) + "\n",
          f"New features: {R.EXTRA}. Candidates: top-50 of {NEW_COMPS} + top-50 popularity (old = Phase H rule).\n",
          "Candidate recall and size:\n", rec.to_markdown(index=False, floatfmt=".4f"), "",
          "v2 grid (alone, ranked by mean):\n", g.drop(columns=["_g", "_cfg"]).to_markdown(index=False, floatfmt=".5f"), "",
          "Comparison (paired bootstrap vs h02-equivalent, 1000 resamples, seed 0):\n",
          comp.to_markdown(index=False, floatfmt=".5f"), "",
          "Ablation (fixed lr 0.05 / 15 leaves / 200 iterations; 'old' windows = Phase H training windows):\n",
          abl.to_markdown(index=False, floatfmt=".5f"), "",
          f"Files (refit on {'+'.join(used['test'])}, validated; {time.time() - t:.1f}s):\n",
          man.to_markdown(index=False), "", note, "",
          f"Phase J runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseJ.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
