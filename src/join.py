"""Phase A step 3: join view intervals to schedule airings.

Rule: a view (user, channel, [start, stop]) is matched to every airing on the SAME channel whose
interval [start_time, start_time + duration] overlaps it; overlap seconds =
min(stops) - max(starts) > 0. The week of a pair is the ISO week of the VIEW start_time.

`airing_interactions()` aggregates to one row per (user, airing, week) and caches it; the
"watched" threshold is applied later (see `watched()`), so it stays a tunable parameter.
"""
import time

import numpy as np
import pandas as pd

import data

ITX_CACHE = data.CACHE / "airing_itx.parquet"


def schedule_for(parts) -> pd.DataFrame:
    """Concatenated schedule with a global airing_id (row order = channel, start)."""
    s = data.load_schedule_all(parts).sort_values(["channel_id", "start_time"], kind="stable")
    s = s.reset_index(drop=True)
    s["airing_id"] = np.arange(len(s), dtype=np.int64)
    return s


def overlap_pairs(views: pd.DataFrame, sched: pd.DataFrame, shift_s: int = 0) -> pd.DataFrame:
    """All (view_idx, airing_id, overlap_s) with overlap > 0 on the same channel.

    views: columns vsetv_id, start_time, stop_time (index = view idx).
    shift_s: added to view timestamps (only used for the time-alignment check).
    """
    vs = views.start_time.values.astype("datetime64[s]").astype(np.int64) + shift_s
    ve = views.stop_time.values.astype("datetime64[s]").astype(np.int64) + shift_s
    vch = views.vsetv_id.values
    vidx = views.index.values
    a_st = sched.start_time.values.astype("datetime64[s]").astype(np.int64)
    a_en = sched.end_time.values.astype("datetime64[s]").astype(np.int64)
    a_ch = sched.channel_id.values
    out = []
    # channel blocks in the (channel, start)-sorted schedule
    ch_u, ch_first, ch_cnt = np.unique(a_ch, return_index=True, return_counts=True)
    order = np.argsort(vch, kind="stable")
    v_ch_sorted = vch[order]
    for ch, f, c in zip(ch_u, ch_first, ch_cnt):
        lo, hi = np.searchsorted(v_ch_sorted, [ch, ch + 1])
        if lo == hi:
            continue
        sel = order[lo:hi]
        st, en = a_st[f:f + c], a_en[f:f + c]
        en_cm = np.maximum.accumulate(en)  # monotone; robust to the few overlapping airings
        i0 = np.searchsorted(en_cm, vs[sel], side="right")  # first airing that may end after view start
        i1 = np.searchsorted(st, ve[sel], side="left")      # airings starting before view stop
        n = np.maximum(i1 - i0, 0)
        if n.sum() == 0:
            continue
        rep = np.repeat(np.arange(len(sel)), n)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ai = i0[rep] + offs
        ov = np.minimum(ve[sel][rep], en[ai]) - np.maximum(vs[sel][rep], st[ai])
        keep = ov > 0
        out.append(pd.DataFrame({"view_idx": vidx[sel][rep][keep],
                                 "airing_id": (f + ai[keep]).astype(np.int64),
                                 "overlap_s": ov[keep].astype(np.int64)}))
    return pd.concat(out, ignore_index=True)


def airing_interactions(force=False) -> pd.DataFrame:
    """(user, airing, week) rows with summed overlap, for ALL airings incl. tv_show_id == 0.
    Built from weeks 11-30 log x 11-20 + 21-30 schedules."""
    if ITX_CACHE.exists() and not force:
        return pd.read_parquet(ITX_CACHE)
    log = data.load_log().drop_duplicates()
    sched = schedule_for(("11-20", "21-30"))
    p = overlap_pairs(log, sched)
    p["user_id"] = log.user_id.values[log.index.get_indexer(p.view_idx)]
    p["week"] = log.week.values[log.index.get_indexer(p.view_idx)]
    g = p.groupby(["user_id", "airing_id", "week"], as_index=False).overlap_s.sum()
    g = g.merge(sched[["airing_id", "channel_id", "tv_show_id", "start_time", "duration"]], on="airing_id")
    # overlapping duplicate view rows could double count; cap at airing length
    g["overlap_s"] = np.minimum(g.overlap_s, g.duration)
    g["frac"] = g.overlap_s / g.duration.clip(lower=1)
    g.to_parquet(ITX_CACHE)
    return g


def watched(itx: pd.DataFrame, min_sec: int = 0, min_frac: float = 0.0) -> pd.DataFrame:
    """Labelled airings counted as watched: overlap >= min_sec OR frac >= min_frac
    (with min_frac == 0 only the absolute rule applies)."""
    lab = itx[itx.tv_show_id != 0]
    ok = lab.overlap_s >= min_sec
    if min_frac > 0:
        ok = ok | (lab.frac >= min_frac)
    return lab[ok]


def user_show_week(w: pd.DataFrame) -> pd.DataFrame:
    return w.groupby(["user_id", "tv_show_id", "week"], as_index=False).agg(
        sec=("overlap_s", "sum"), n_airings=("airing_id", "nunique"))
