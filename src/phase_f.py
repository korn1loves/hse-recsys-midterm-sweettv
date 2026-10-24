"""Phase F: fixed user-specified configs with several habit components at once (no tuning).
Base = pers 1, rate 0.25, AR 0.25, MF 1, min-max fusion. Validation numbers are for the record only.
Writes results/submission_f01..f08.csv, results/phaseF_manifest.csv, results/phaseF.md."""
import time

import pandas as pd

import baselines as B
import data
import join
import phase_d as PD
import scoring as SC
from metrics import mapk
from phase_e import BASE, VP, components, fuse

RES = data.ROOT / "results"
CONFIGS = {
    "f01": {"daypart_4": 1.5},
    "f02": {"daypart_4": 2.0},
    "f03": {"daypart_4": 1.0, "channel_4": 0.5},
    "f04": {"daypart_4": 1.0, "channel_4": 1.0},
    "f05": {"daypart_4": 1.0, "slot_4": 0.5},
    "f06": {"daypart_None": 1.0},
    "f07": {"daypart_4": 1.0, "channel_4": 0.5, "slot_4": 0.5},
    "f08": {"channel_4": 1.0},
}


def overlap(a, b):
    x = pd.read_csv(a, dtype=str).tv_show_id.str.split()
    y = pd.read_csv(b, dtype=str).tv_show_id.str.split()
    return sum(len(set(p) & set(q)) for p, q in zip(x, y)) / (5 * len(x))


def main():
    T0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    val = {}
    for s, (hw, tw) in B.SPLITS.items():
        sp = B.make_split(s, itx, 0, users)
        fb = B.popularity(sp)
        comps, _ = components(sp, list(hw), list(tw), VP, VP)
        for name, extra in CONFIGS.items():
            val[(name, s)] = mapk(sp.target, SC.topk(*fuse(comps, {**BASE, **extra}), sp, fb))

    hist = join.user_show_week(join.watched(itx, min_sec=0))
    hist = hist[hist.week.between(11, 30)]
    pool = B.pool_from_schedule(("31-42",), range(31, 43))
    spf = B.Split("final", hist, pool, users, {}, 30)
    cf, _ = components(spf, list(range(11, 31)), list(range(31, 43)), VP, ("31-42",))
    fbf = B.popularity(spf)
    rows = []
    for name, extra in CONFIGS.items():
        w = {**BASE, **extra}
        p = SC.topk(*fuse(cf, w), spf, fbf)
        B.check_preds(p, users, pool)
        fn = f"submission_{name}.csv"
        PD.write_sub(p, pool, fn)
        m, s2 = val[(name, "main")], val[(name, "secondary")]
        rows.append({"file": f"results/{fn}", "habits": " + ".join(f"{k} {v:g}" for k, v in extra.items()),
                     "weights": " ".join(f"{k}={v:g}" for k, v in w.items()), "fusion": "minmax",
                     "main": round(m, 5), "secondary": round(s2, 5), "mean": round((m + s2) / 2, 5),
                     "overlap_with_e04": round(overlap(RES / fn, RES / "submission_e04.csv"), 3)})
    man = pd.DataFrame(rows)
    man.to_csv(RES / "phaseF_manifest.csv", index=False)
    md = man.to_markdown(index=False)
    (RES / "phaseF.md").write_text(md + f"\n\nPhase F runtime {time.time() - T0:.1f}s\n")
    print(md, f"\nruntime {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
