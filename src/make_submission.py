"""Phase A step 7: retrain the best reference (d_personal, history threshold 0 s) on weeks 11-30,
candidate pool from the 31-42 schedule, write results/submission_baseline.csv and validate it."""
import sys
import time

import pandas as pd

import baselines as B
import data
import join

HIST_MIN_SEC = 0
METHOD = "d_personal"


def write_submission(preds: dict, path):
    sub = data.load_submission()
    body = "\n".join(f"{u},{' '.join(map(str, preds[u]))}" for u in sub.user_id)
    path.write_text("user_id,tv_show_id\n" + body)  # template has LF and no trailing newline


def validate(path, pool):
    tmpl = (data.DATA / "submission.csv").read_bytes().split(b"\n")
    got = path.read_bytes().split(b"\n")
    assert got[0] == tmpl[0], "header"
    assert len(got) == len(tmpl) == 1261, (len(got), len(tmpl))
    pool = set(pool)
    for a, b in zip(tmpl[1:], got[1:]):
        ua, _ = a.decode().split(",")
        ub, ids = b.decode().split(",")
        assert ua == ub, "user order"
        ids = [int(x) for x in ids.split(" ")]
        assert len(ids) == 5 and len(set(ids)) == 5 and 0 not in ids and set(ids) <= pool, ub
    df = pd.read_csv(path, dtype=str)
    assert list(df.columns) == ["user_id", "tv_show_id"] and len(df) == 1260
    return True


def main():
    t0 = time.time()
    itx = join.airing_interactions()
    users = data.load_submission().user_id.tolist()
    hist = join.user_show_week(join.watched(itx, min_sec=HIST_MIN_SEC))
    hist = hist[hist.week.between(11, 30)]
    pool = B.pool_from_schedule(("31-42",), range(31, 43))
    sp = B.Split("final", hist, pool, users, {}, 30)
    preds = B.methods(sp)[METHOD]()
    B.check_preds(preds, users, pool)
    out = data.ROOT / "results" / "submission_baseline.csv"
    write_submission(preds, out)
    validate(out, pool)
    pers_users = set(hist.user_id[hist.tv_show_id.isin(pool)])
    n_pers = sum(1 for u in users if u in pers_users)
    print(f"pool {len(pool):,}; users with in-pool history {n_pers}/{len(users)}; "
          f"fallback top-5 {B.popularity(sp)[:5]}; validated OK; runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
