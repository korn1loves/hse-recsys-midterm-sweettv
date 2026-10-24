"""Phase H2: transfer-robust re-rankers without exact-hour slot signals.
Variant A ("no-slot features"): drop exp_slot_pw and rank_slot; candidates unchanged (incl. slot_4 top-30).
Variant B ("no-slot features + candidates"): additionally drop slot_4 from candidate generation.
Same windows, grid and leak-free evaluation as Phase H. Files: submission_h04 (A), h05 (B),
h06 (rank-average blend A / f03 0.5/0.5). Writes results/phaseH2.md, appends to results/phaseH_manifest.csv."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import data
import join
import phase_d as PD
import rerank as R
import scoring as SC
from cbf import item_table
from phase_h import GRID, WINDOWS

RES = data.ROOT / "results"
DROP = ["exp_slot_pw", "rank_slot"]
KEEP = [i for i, f in enumerate(R.FEATURES) if f not in DROP]
FEATS = [R.FEATURES[i] for i in KEEP]
CAT = FEATS.index("category")
H01 = (0.05, 15, 200)


def prep(X, cols):
    X = X[:, cols].copy()
    ci = [R.FEATURES[i] for i in cols].index("category")
    X[:, ci] += 1
    return X, ci


def fit(ws, cfg, cols):
    X, ci = prep(np.vstack([w["X"][w["train_mask"]] for w in ws]), cols)
    y = np.concatenate([w["y"][w["train_mask"]] for w in ws])
    cat = np.zeros(X.shape[1], dtype=bool)
    cat[ci] = True
    m = HistGradientBoostingClassifier(learning_rate=cfg[0], max_leaf_nodes=cfg[1], max_iter=cfg[2],
                                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                       categorical_features=cat, random_state=0)
    t = time.time()
    m.fit(X, y)
    m.fit_s, m.cols = time.time() - t, cols
    return m


def predict(m, w, X=None):
    Xp, _ = prep(w["X"] if X is None else X, m.cols)
    return R.preds_from_scores(w, m.predict_proba(Xp)[:, 1])


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
    WA, WB, rec = {}, {}, []
    for k, (hw, tw) in WINDOWS.items():
        R.CAND_COMPS = ["pers", "rate", "daypart_4", "channel_4", "slot_4", "ar", "mf"]
        WA[k] = R.build(k, itx, hw, tw, users, cats)
        R.CAND_COMPS = ["pers", "rate", "daypart_4", "channel_4", "ar", "mf"]
        WB[k] = R.build(k, itx, hw, tw, users, cats)
        rec.append({"window": k, "recall (with slot cands)": WA[k]["recall"], "avg cands (with slot)": WA[k]["avg_cand"],
                    "recall (no slot cands)": WB[k]["recall"], "avg cands (no slot)": WB[k]["avg_cand"]})
        print(rec[-1])
    allcols = list(range(len(R.FEATURES)))
    ev = {"main": (["W0"], "W1"), "secondary": (["W0", "W1b"], "W2")}
    ref = {}
    for s, (tr, te) in ev.items():
        w = WA[te]
        ref[(s, "f03")] = apv(w, SC.topk(*w["f03"], w["sp"], w["fb"]))
        ref[(s, "h01")] = apv(w, predict(fit([WA[k] for k in tr], H01, allcols), w)[0])

    variants = {"A: no-slot features": WA, "B: no-slot features + no slot candidates": WB}
    grid_rows, best, models = [], {}, {}
    for vname, Wv in variants.items():
        rows = []
        for cfg in GRID:
            r = {"variant": vname, "learning_rate": cfg[0], "max_leaf_nodes": cfg[1], "max_iter": cfg[2]}
            for s, (tr, te) in ev.items():
                m = fit([Wv[k] for k in tr], cfg, KEEP)
                r[s] = apv(Wv[te], predict(m, Wv[te])[0]).mean()
                models[(vname, cfg, s)] = m
            r["mean"] = (r["main"] + r["secondary"]) / 2
            rows.append(r)
        rows = sorted(rows, key=lambda x: -x["mean"])
        best[vname] = (rows[0]["learning_rate"], int(rows[0]["max_leaf_nodes"]), int(rows[0]["max_iter"]))
        grid_rows += rows
    grid = pd.DataFrame(grid_rows)

    comp, imps = [], {}
    for vname, Wv in variants.items():
        for s, (tr, te) in ev.items():
            w = Wv[te]
            m = models[(vname, best[vname], s)]
            p, S, V = predict(m, w)
            a = apv(w, p)
            comp.append({"split": s, "method": f"Re-ranker {vname}", "MAP@5": a.mean(),
                         "Δ vs f03": boot(a, ref[(s, "f03")]), "Δ vs h01": boot(a, ref[(s, "h01")])})
            if vname.startswith("A"):
                ab = apv(w, R.blend(w, S, V))
                comp.append({"split": s, "method": "Blend A / f03 0.5/0.5 (h06)", "MAP@5": ab.mean(),
                             "Δ vs f03": boot(ab, ref[(s, "f03")]), "Δ vs h01": boot(ab, ref[(s, "h01")])})
        # permutation importance on main
        w, m = Wv["W1"], models[(vname, best[vname], "main")]
        base = apv(w, predict(m, w)[0]).mean()
        rng = np.random.default_rng(0)
        imp = []
        for j in KEEP:
            drops = []
            for _ in range(3):
                X = w["X"].copy()
                X[:, j] = X[rng.permutation(len(X)), j]
                drops.append(base - apv(w, predict(m, w, X)[0]).mean())
            imp.append({"feature": R.FEATURES[j], "MAP@5 drop": np.mean(drops)})
        imps[vname] = pd.DataFrame(imp).sort_values("MAP@5 drop", ascending=False).head(10)
    for s in ev:
        comp += [{"split": s, "method": "h01 (re-ranker with slot)", "MAP@5": ref[(s, "h01")].mean(),
                  "Δ vs f03": boot(ref[(s, "h01")], ref[(s, "f03")]), "Δ vs h01": "—"},
                 {"split": s, "method": "f03", "MAP@5": ref[(s, "f03")].mean(), "Δ vs f03": "—", "Δ vs h01": "—"}]
    comp = pd.DataFrame(comp).sort_values(["split", "MAP@5"], ascending=[True, False])

    # final files
    t = time.time()
    mA = fit([WA["W0"], WA["W1"], WA["W2"]], best["A: no-slot features"], KEEP)
    mB = fit([WB["W0"], WB["W1"], WB["W2"]], best["B: no-slot features + no slot candidates"], KEEP)
    pA, SA, VA = predict(mA, WA["TEST"])
    pB, _, _ = predict(mB, WB["TEST"])
    pC = R.blend(WA["TEST"], SA, VA)
    pool = np.asarray(WA["TEST"]["sp"].pool)
    f03 = pd.read_csv(RES / "submission_f03.csv", dtype=str).tv_show_id.str.split()
    h01 = pd.read_csv(RES / "submission_h01.csv", dtype=str).tv_show_id.str.split()
    ov = lambda x, y: round(sum(len(set(a) & set(b)) for a, b in zip(x, y)) / (5 * len(x)), 3)
    val = lambda meth, s: round(comp[(comp.split == s) & (comp.method == meth)]["MAP@5"].iloc[0], 5)
    man = []
    for fn, p, desc, vname, meth in [
            ("submission_h04.csv", pA, "re-ranker, no slot features", "A: no-slot features", "Re-ranker A: no-slot features"),
            ("submission_h05.csv", pB, "re-ranker, no slot features, no slot candidates",
             "B: no-slot features + no slot candidates", "Re-ranker B: no-slot features + no slot candidates"),
            ("submission_h06.csv", pC, "rank-average blend h04 / f03 0.5/0.5", "A: no-slot features",
             "Blend A / f03 0.5/0.5 (h06)")]:
        PD.write_sub(p, pool, fn)
        x = pd.read_csv(RES / fn, dtype=str).tv_show_id.str.split()
        b = best[vname]
        man.append({"file": f"results/{fn}", "method": desc, "trained_on": "W0+W1+W2",
                    "params": f"lr={b[0]} leaves={b[1]} max_iter={b[2]} (early stopping)",
                    "main": val(meth, "main"), "secondary": val(meth, "secondary"),
                    "overlap_with_f03": ov(x, f03), "overlap_with_h01": ov(x, h01)})
    man = pd.DataFrame(man)
    old = pd.read_csv(RES / "phaseH_manifest.csv")
    old = old[~old.file.isin(man.file)]
    pd.concat([old, man], ignore_index=True).to_csv(RES / "phaseH_manifest.csv", index=False)

    md = ["### Phase H2: transfer-robust re-rankers without exact-hour slot signals (generated by src/phase_h2.py)\n",
          f"Dropped features: {DROP} ({len(FEATS)} features left). Variant B also removes slot_4 from candidate "
          "generation. Same windows, grid, leak-free training (main: W0; secondary: W0+W1b; final: W0+W1+W2).\n",
          "Candidate recall:\n", pd.DataFrame(rec).to_markdown(index=False, floatfmt=".4f"), "",
          "Grid (ranked by mean within variant):\n", grid.to_markdown(index=False, floatfmt=".5f"), "",
          f"Best: {best}\n", "Comparison (paired bootstrap, 1000 resamples, seed 0):\n",
          comp.to_markdown(index=False, floatfmt=".5f"), ""]
    for vname, imp in imps.items():
        md += [f"Permutation importance on main, {vname} (top 10):\n", imp.to_markdown(index=False, floatfmt=".5f"), ""]
    md += [f"Files (validated; {time.time() - t:.1f}s):\n", man.to_markdown(index=False), "",
           f"Phase H2 runtime {time.time() - T0:.1f}s\n"]
    (RES / "phaseH2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
