"""Phase D methods.

Schedule-slot CBF ("exposure"): slot = (channel, hour-of-week), 168 hours per channel.
- Item side A[slot, show]: seconds of the show's labelled airings (airing START in a target week)
  overlapping each clock hour, summed over the target window (schedule only).
- User side P[user, slot]: seconds the user watched channel c in hour-of-week h over the history
  weeks (ALL views, labelled or not; views clipped at the start of the target window), divided by
  (n_history_weeks * 3600) -> probability of watching that slot. With a recency half-life, week w
  gets weight 0.5^((last_week - w)/hl) and the denominator is sum_w weight_w * 3600.
- Score = P @ A = expected watched seconds of the show in the target window.

Phase E coarse habit variants (gran="daypart": (channel, weekday/weekend, 4-hour block), 12 slots
per channel; gran="channel": channel only) use the same construction with coarser slots.

Rate-normalised personal history: rate(u,s) = user's history seconds on s / s's total history
airtime (labelled airings starting in history weeks); score = rate * s's target-window airtime.
"""
import datetime as dt
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps

import data

HOURS = 168
# slots per channel for each granularity: exact hour-of-week, (weekday/weekend x 4-hour block), channel only
GRAN = {"slot": 168, "daypart": 12, "channel": 1}


def _slot(ch, hour, gran="slot"):
    """Slot index for channel ids and epoch-hour indices at the given granularity."""
    how = _how(hour)
    if gran == "slot":
        sub = how
    elif gran == "daypart":
        sub = (how // 24 >= 5).astype(np.int64) * 6 + (how % 24) // 4
    else:
        sub = np.zeros_like(how)
    return ch.astype(np.int64) * GRAN[gran] + sub


def _hour_pieces(start_s, stop_s):
    """Split [start, stop) (epoch seconds) into clock-hour pieces -> (row index, hour index, seconds)."""
    h0 = start_s // 3600
    h1 = (stop_s - 1) // 3600
    n = np.maximum(h1 - h0 + 1, 0)
    rep = np.repeat(np.arange(len(start_s)), n)
    hour = h0[rep] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    sec = np.minimum(stop_s[rep], (hour + 1) * 3600) - np.maximum(start_s[rep], hour * 3600)
    return rep, hour, sec


def _how(hour):
    """hour-of-week (Mon 00:00 = 0) from epoch hour index (1970-01-01 was a Thursday)."""
    return ((hour // 24 + 3) % 7) * 24 + hour % 24


def _epoch(ts):
    return ts.values.astype("datetime64[s]").astype(np.int64)


def week_start_epoch(week, year=2020):
    return int(dt.datetime.combine(dt.date.fromisocalendar(year, week, 1), dt.time()).timestamp()
               - dt.datetime(1970, 1, 1).timestamp())


def airtime_matrix(parts, weeks, pool, n_ch, gran="slot", per_airing=False):
    """A: (n_ch*168) x len(pool) seconds of labelled airtime per slot, airings starting in `weeks`."""
    s = data.load_schedule_all(parts)
    s = s[(s.tv_show_id != 0) & s.week.isin(list(weeks)) & s.tv_show_id.isin(pool)]
    st = _epoch(s.start_time)
    rep, hour, sec = _hour_pieces(st, st + s.duration.values)
    slot = _slot(s.channel_id.values[rep], hour, gran)
    col = np.searchsorted(pool, s.tv_show_id.values[rep])
    if per_airing:  # Phase M: each airing contributes 1 in total, split over its slots by time share
        sec = sec / np.maximum(s.duration.values[rep], 1)
    A = sps.csr_matrix((sec.astype(float), (slot, col)), shape=(n_ch * GRAN[gran], len(pool)))
    A.sum_duplicates()
    return A


def show_airtime(parts, weeks):
    s = data.load_schedule_all(parts)
    s = s[(s.tv_show_id != 0) & s.week.isin(list(weeks))]
    return s.groupby("tv_show_id").duration.sum()


def profile_matrix(users, hist_weeks, n_ch, half_life=None, gran="slot"):
    """P: len(users) x (n_ch*168) slot-watch probabilities from all history views."""
    log = data.load_log().drop_duplicates()
    log = log[log.user_id.isin(users) & log.week.isin(list(hist_weeks))]
    st = _epoch(log.start_time)
    en = np.minimum(_epoch(log.stop_time), week_start_epoch(max(hist_weeks) + 1))  # no target-window seconds
    rep, hour, sec = _hour_pieces(st, en)
    wk = log.week.values[rep]
    ref = max(hist_weeks)
    if half_life is None:
        w = np.ones(len(sec))
        denom = len(hist_weeks) * 3600.0
    else:
        w = 0.5 ** ((ref - wk) / half_life)
        denom = sum(0.5 ** ((ref - x) / half_life) for x in hist_weeks) * 3600.0
    users = np.asarray(users, dtype=np.uint64)
    order = np.argsort(users)
    row = order[np.searchsorted(users[order], log.user_id.values[rep])]
    slot = _slot(log.vsetv_id.values[rep], hour, gran)
    P = sps.csr_matrix((sec * w / denom, (row, slot)), shape=(len(users), n_ch * GRAN[gran]))
    P.sum_duplicates()
    return P


def n_channels(parts):
    return int(max(data.load_log().vsetv_id.max(), data.load_schedule_all(parts).channel_id.max())) + 1


def exposure_scores(sp, hist_weeks, sched_parts, target_weeks, half_life=None, gran="slot", per_airing=False):
    t0 = time.time()
    n_ch = n_channels(sched_parts)
    P = profile_matrix(sp.users, hist_weeks, n_ch, half_life, gran)
    A = airtime_matrix(sched_parts, target_weeks, np.asarray(sp.pool), n_ch, gran, per_airing)
    fit = time.time() - t0
    t1 = time.time()
    S = (P @ A).toarray()
    return S, S > 0, {"fit_s": fit, "predict_s": time.time() - t1}


def rate_scores(sp, hist_parts, hist_weeks, target_parts, target_weeks):
    t0 = time.time()
    h = sp.hist[sp.hist.user_id.isin(sp.users)].groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    hair = show_airtime(hist_parts, hist_weeks)
    tair = show_airtime(target_parts, target_weeks).reindex(sp.pool).fillna(0).values
    h = h[h.tv_show_id.isin(sp.pool) & h.tv_show_id.isin(hair.index)]
    rate = h.sec.values / hair.reindex(h.tv_show_id).values
    users = np.asarray(sp.users, dtype=np.uint64)
    order = np.argsort(users)
    r = order[np.searchsorted(users[order], h.user_id.values)]
    c = np.searchsorted(np.asarray(sp.pool), h.tv_show_id.values)
    S = np.zeros((len(users), len(sp.pool)))
    np.add.at(S, (r, c), rate)
    S *= tair[None, :]
    return S, S > 0, {"fit_s": time.time() - t0, "predict_s": 0.0}


def show_airings(parts, weeks):
    """Number of labelled airings per show starting in `weeks` (Phase M episode units)."""
    s = data.load_schedule_all(parts)
    s = s[(s.tv_show_id != 0) & s.week.isin(list(weeks))]
    return s.groupby("tv_show_id").size()


def rate_ep_scores(sp, hist_parts, hist_weeks, target_parts, target_weeks):
    """Episode rate: user's history episodes on s / s's history airings x s's target-window airings
    (sp.hist["sec"] holds episodes in Phase M)."""
    t0 = time.time()
    h = sp.hist[sp.hist.user_id.isin(sp.users)].groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    hair = show_airings(hist_parts, hist_weeks)
    tair = show_airings(target_parts, target_weeks).reindex(sp.pool).fillna(0).values
    h = h[h.tv_show_id.isin(sp.pool) & h.tv_show_id.isin(hair.index)]
    rate = h.sec.values / hair.reindex(h.tv_show_id).values
    users = np.asarray(sp.users, dtype=np.uint64)
    order = np.argsort(users)
    r = order[np.searchsorted(users[order], h.user_id.values)]
    c = np.searchsorted(np.asarray(sp.pool), h.tv_show_id.values)
    S = np.zeros((len(users), len(sp.pool)))
    np.add.at(S, (r, c), rate)
    S *= tair[None, :]
    return S, S > 0, {"fit_s": time.time() - t0, "predict_s": 0.0}
