"""Phase M: episode-equivalent units (target T5 and history).

episode(user, airing) = min(overlap / airing duration, 1), from the airing-level table of Phase L.
History table (for all components): per (user, show, week) the sum of episodes over ALL labelled
airings (the l05 rule), stored in column "sec" so that existing components work in episode units.
Target variants (target window, top-5 per user, ties -> seconds, then smaller id):
  a: watched airings (overlap >= 300 s)  [T5 of Phase L; the target used everywhere in Phase M]
  b: all airings;  c: airings with fraction >= 0.25;  d: airings with overlap >= 60 s.
History ranking variants (personal history): all (l05), w300 (overlap >= 300 s), f25 (fraction >= 0.25),
hl8 (all airings, recency half-life 8 weeks).
"""
import numpy as np
import pandas as pd

import baselines as B
from phase_l import airing_table

_A = {}


def table(itx):
    if id(itx) not in _A:
        _A.clear()
        _A[id(itx)] = airing_table(itx)
    return _A[id(itx)]


def _filter(a, variant):
    if variant in ("all", "b", "hl8"):
        return a
    if variant in ("w300", "a"):
        return a[a.overlap >= 300]
    if variant in ("f25", "c"):
        return a[a.epi >= 0.25]
    if variant == "d":
        return a[a.overlap >= 60]
    raise ValueError(variant)


def ep_hist(itx, variant="all"):
    a = _filter(table(itx), variant)
    return a.groupby(["user_id", "tv_show_id", "week"], as_index=False).agg(sec=("epi", "sum"))


def target_ep(itx, tw, users, variant="a"):
    a = _filter(table(itx), variant)
    a = a[a.week.isin(list(tw)) & a.user_id.isin(users)]
    g = a.groupby(["user_id", "tv_show_id"], as_index=False).agg(score=("epi", "sum"), s=("overlap", "sum"))
    g = g.sort_values(["user_id", "score", "s", "tv_show_id"], ascending=[True, False, False, True])
    return {u: grp.tv_show_id.head(B.K).tolist() for u, grp in g.groupby("user_id")}


def rank_personal(itx, hw, pool, users, variant, fallback):
    a = _filter(table(itx), variant)
    a = a[a.week.isin(list(hw)) & a.user_id.isin(users) & a.tv_show_id.isin(pool)]
    w = 0.5 ** ((max(hw) - a.week) / 8.0) if variant == "hl8" else 1.0
    a = a.assign(e=a.epi * w)
    g = a.groupby(["user_id", "tv_show_id"], as_index=False).agg(score=("e", "sum"), s=("overlap", "sum"))
    g = g[g.score > 0].sort_values(["user_id", "score", "s", "tv_show_id"], ascending=[True, False, False, True])
    per = {u: grp.tv_show_id.head(B.K).tolist() for u, grp in g.groupby("user_id")}
    return {u: B.fill(per.get(u, []), fallback) for u in users}
