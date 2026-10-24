"""Phase H: candidate generation + features for a learned re-ranker.

Window = (history weeks -> target weeks), built exactly like the existing splits (history = all
labelled overlap in history weeks; target = top-5 watched (>= 300 s) labelled shows in target weeks;
pool = labelled shows airing in the target window). Only the 1,260 submission users get candidates;
components that need a population (AR, MF, popularity) are fitted on all log users' history, as before.

Candidates: union of the top 30 valid pool items of each component (pers, rate, daypart_4, channel_4,
slot_4, AR, MF) plus the top 30 of pool popularity.

Scale handling: windows have different lengths, so seconds-type features are divided by the number of
history weeks (history quantities) or target weeks (target-window airtime and exposure).
"""
import numpy as np
import pandas as pd

import baselines as B
import data
import join
import phase_e as PE
import scoring as SC
from cbf import item_table
from exposure import show_airtime
from phase_e import fuse
from phase_g import F03

PE.HABITS = ["slot_4", "daypart_4", "channel_4"]
VP = ("11-20", "21-30")
TOPN = 30
CAND_COMPS = ["pers", "rate", "daypart_4", "channel_4", "slot_4", "ar", "mf"]
FEATURES = ["pers_sec_pw", "pers_last4_sec", "n_weeks", "weeks_since_last", "rate",
            "show_hist_air_pw", "show_tgt_air_pw", "air_ratio", "exp_slot_pw", "exp_daypart_pw",
            "exp_channel_pw", "ar", "mf", "pop_sec_pw", "pop_users", "category", "is_new_id",
            "user_tot_sec_pw", "user_n_shows", "user_cat_share",
            "rank_pers", "rank_rate", "rank_daypart", "rank_channel", "rank_slot"]
CAT_IDX = FEATURES.index("category")
# Phase J extra features (appended after FEATURES when build(extra=True))
EXTRA = ["air_trend", "prime_share_show", "premiere_recent", "premiere_target", "user_prime_share"]
OLD_COMPS = ["pers", "rate", "daypart_4", "channel_4", "slot_4", "ar", "mf"]
_FIRST_WEEK = None


def first_airing_week():
    """ISO week of each labelled show's first airing over all three schedules (schedule metadata only)."""
    global _FIRST_WEEK
    if _FIRST_WEEK is None:
        s = data.load_schedule_all(("11-20", "21-30", "31-42"))
        s = s[s.tv_show_id != 0]
        _FIRST_WEEK = s.groupby("tv_show_id").week.min()
    return _FIRST_WEEK


def _prime(start_s, stop_s):
    """Seconds of [start, stop) falling in 19:00-23:00 and total seconds (via clock-hour pieces)."""
    from exposure import _hour_pieces
    rep, hour, sec = _hour_pieces(start_s, stop_s)
    prime = (hour % 24 >= 19) & (hour % 24 < 23)
    return rep, sec, prime


def extra_features(sp, hw, tw, tparts, users, c, r):
    from exposure import _epoch, week_start_epoch
    pool = np.asarray(sp.pool)
    last4 = [w for w in hw if w > max(hw) - 4]
    a4 = show_airtime(VP, last4).reindex(pool).fillna(0).values / len(last4)
    tair = show_airtime(tparts, tw).reindex(pool).fillna(0).values / len(tw)
    trend = tair / (a4 + 1.0)
    s = data.load_schedule_all(tparts)
    s = s[(s.tv_show_id != 0) & s.week.isin(tw) & s.tv_show_id.isin(pool)]
    st = _epoch(s.start_time)
    rep, sec, prime = _prime(st, st + s.duration.values)
    col = np.searchsorted(pool, s.tv_show_id.values[rep])
    tot = np.bincount(col, sec, len(pool))
    pr = np.bincount(col, sec * prime, len(pool))
    pshare = np.where(tot > 0, pr / np.maximum(tot, 1), 0)
    fw = first_airing_week().reindex(pool).values
    prem_recent = ((fw > max(hw) - 4) & (fw <= max(hw))).astype(float)
    prem_target = (fw >= min(tw)).astype(float)
    log = data.load_log().drop_duplicates()
    log = log[log.user_id.isin(users) & log.week.isin(hw)]
    ls = _epoch(log.start_time)
    le = np.minimum(_epoch(log.stop_time), week_start_epoch(max(hw) + 1))
    lrep, lsec, lprime = _prime(ls, le)
    ur = _rows(users, log.user_id.values[lrep])
    utot = np.bincount(ur, lsec, len(users))
    upr = np.bincount(ur, lsec * lprime, len(users))
    ushare = np.where(utot > 0, upr / np.maximum(utot, 1), 0)
    return np.column_stack([trend[c], pshare[c], prem_recent[c], prem_target[c], ushare[r]]).astype(np.float32)


def make_window(name, itx, hw, tw, users):
    hw, tw = list(hw), list(tw)
    tparts = ("31-42",) if min(tw) >= 31 else VP
    hist = join.user_show_week(join.watched(itx, min_sec=0))
    hist = hist[hist.week.isin(hw)]
    pool = B.pool_from_schedule(tparts, tw)
    target = B.make_target(itx, tw, users) if max(tw) <= 30 else {}
    sp = B.Split(name, hist, pool, users, target, max(hw))
    return sp, hw, tw, tparts


def rank_rows(S, valid):
    """Within-user rank (1 = best, ties -> smaller id); invalid items get len(pool) + 1."""
    M = np.where(valid, S, -np.inf)
    order = np.argsort(-M, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(M.shape[0])[:, None], order] = np.arange(1, M.shape[1] + 1)[None, :]
    return np.where(valid, ranks, M.shape[1] + 1)


def _rows(users, uids):
    users = np.asarray(users, dtype=np.uint64)
    o = np.argsort(users)
    return o[np.searchsorted(users[o], np.asarray(uids, dtype=np.uint64))]


def build(name, itx, hw, tw, users, cats, fusions=None, topn=None, cand_comps=None, extra=False):
    """Returns dict with candidates (r, c), features X, labels y, f03 scores, fallback, split.
    fusions: optional {name: weights} of extra min-max hybrids to return as (S, V) under "fus".
    topn / cand_comps override TOPN / CAND_COMPS; rows are flagged "old_cand" if they would also be
    candidates under the Phase H rule (top-30 of OLD_COMPS + top-30 popularity).
    extra=True appends the EXTRA features; "yg" = graded label 5..1 by target rank (0 otherwise)."""
    topn = topn or TOPN
    cand_comps = cand_comps or CAND_COMPS
    sp, hw, tw, tparts = make_window(name, itx, hw, tw, users)
    comps, _ = PE.components(sp, hw, tw, VP, tparts)
    pool = np.asarray(sp.pool)
    n_u, n_i = len(sp.users), len(pool)
    nh, nt = len(hw), len(tw)
    fb = B.popularity(sp)

    # candidates
    def cand_mask(comp_names, n):
        cm = np.zeros((n_u, n_i), dtype=bool)
        for k in comp_names:
            S, v = comps[k]
            M = np.where(v, S, -np.inf)
            top = np.argpartition(-M, n, axis=1)[:, :n]
            sel = np.take_along_axis(v, top, axis=1)
            cm[np.repeat(np.arange(n_u), n)[sel.ravel()], top.ravel()[sel.ravel()]] = True
        cm[:, np.searchsorted(pool, fb[:n])] = True
        return cm

    cand = cand_mask(cand_comps, topn)
    old = cand_mask(OLD_COMPS, 30)
    cand |= old if (topn >= 30 and set(OLD_COMPS) <= set(cand_comps)) else False
    r, c = np.nonzero(cand)
    old_cand = old[r, c]

    # user x pool history statistics (submission users)
    h = sp.hist[sp.hist.user_id.isin(users) & sp.hist.tv_show_id.isin(pool)]
    hr, hc = _rows(users, h.user_id.values), np.searchsorted(pool, h.tv_show_id.values)
    last4 = np.zeros((n_u, n_i)); nweeks = np.zeros((n_u, n_i)); lastw = np.zeros((n_u, n_i))
    rec = h.week.values > sp.ref_week - 4
    np.add.at(last4, (hr[rec], hc[rec]), h.sec.values[rec])
    np.add.at(nweeks, (hr, hc), 1)
    np.maximum.at(lastw, (hr, hc), h.week.values)
    since = np.where(nweeks > 0, sp.ref_week - lastw, 99)

    hair = show_airtime(VP, hw).reindex(pool).fillna(0).values / nh
    tair = show_airtime(tparts, tw).reindex(pool).fillna(0).values / nt
    allh = sp.hist.groupby("tv_show_id").agg(sec=("sec", "sum"), users=("user_id", "nunique"))
    pop_sec = allh.sec.reindex(pool).fillna(0).values / nh
    pop_users = allh.users.reindex(pool).fillna(0).values
    cat_code = cats.reindex(pool).fillna(-1).values
    s = data.load_schedule_all(VP)
    hist_ids = set(s.tv_show_id[(s.tv_show_id != 0) & s.week.isin(hw)])
    is_new = np.array([i not in hist_ids for i in pool], dtype=float)

    uh = sp.hist[sp.hist.user_id.isin(users)]
    utot = np.zeros(n_u); unsh = np.zeros(n_u)
    ur = _rows(users, uh.user_id.values)
    np.add.at(utot, ur, uh.sec.values)
    ush = uh.groupby("user_id").tv_show_id.nunique()
    unsh[_rows(users, ush.index.values)] = ush.values
    ncat = int(cats.max()) + 2
    ucat = np.zeros((n_u, ncat))
    np.add.at(ucat, (ur, cats.reindex(uh.tv_show_id).fillna(-1).values.astype(int) + 1), uh.sec.values)

    ranks = {k: rank_rows(*comps[k]) for k in ["pers", "rate", "daypart_4", "channel_4", "slot_4"]}
    X = np.column_stack([
        comps["pers"][0][r, c] / nh, last4[r, c], nweeks[r, c], since[r, c], comps["rate"][0][r, c],
        hair[c], tair[c], tair[c] / (hair[c] + 1.0),
        comps["slot_4"][0][r, c] / nt, comps["daypart_4"][0][r, c] / nt, comps["channel_4"][0][r, c] / nt,
        comps["ar"][0][r, c], comps["mf"][0][r, c], pop_sec[c], pop_users[c], cat_code[c], is_new[c],
        utot[r] / nh, unsh[r], ucat[r, cat_code[c].astype(int) + 1] / np.maximum(utot[r], 1),
        ranks["pers"][r, c], ranks["rate"][r, c], ranks["daypart_4"][r, c], ranks["channel_4"][r, c],
        ranks["slot_4"][r, c]]).astype(np.float32)

    if extra:
        X = np.hstack([X, extra_features(sp, hw, tw, tparts, users, c, r)])
    grade = {(u, t): 5 - j for u, items in sp.target.items() for j, t in enumerate(items)}
    uarr = np.asarray(sp.users, dtype=np.uint64)
    yg = np.array([grade.get((int(uarr[a]), int(pool[b])), 0) for a, b in zip(r, c)], dtype=np.int8)
    y = (yg > 0).astype(np.int8)
    n_t = sum(len(v) for v in sp.target.values())
    recall = y.sum() / n_t if n_t else float("nan")
    has_t = np.array([u in sp.target for u in sp.users])
    fus = {k: fuse(comps, wts) for k, wts in (fusions or {}).items()}
    return {"sp": sp, "r": r, "c": c, "X": X, "y": y, "yg": yg, "old_cand": old_cand, "tw": tw,
            "fb": fb, "f03": fuse(comps, F03), "fus": fus,
            "recall": recall, "avg_cand": len(r) / n_u, "train_mask": has_t[r]}


def preds_from_scores(w, scores):
    """Top-5 by re-ranker score among the window's candidates, then fallback."""
    sp = w["sp"]
    S = np.full((len(sp.users), len(sp.pool)), -np.inf)
    S[w["r"], w["c"]] = scores
    V = np.isfinite(S)
    return SC.topk(S, V, sp, w["fb"]), S, V


def blend(w, S_rr, V_rr, a=0.5):
    """Rank-average of the re-ranker and f03: score = -(a * rank_rr + (1 - a) * rank_f03)."""
    Sf, Vf = w["f03"]
    R = a * rank_rows(S_rr, V_rr) + (1 - a) * rank_rows(Sf, Vf)
    V = V_rr | Vf
    return SC.topk(-R, V, w["sp"], w["fb"])
