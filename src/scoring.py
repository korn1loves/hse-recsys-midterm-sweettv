"""Shared scoring utilities for Phase B/C: every method returns a dense score matrix
S[len(sp.users), len(sp.pool)] (pool sorted by tv_show_id asc) plus a validity mask; top-5 is
taken among valid items (ties -> smaller tv_show_id) and filled with the Phase A fallback."""
import numpy as np

import baselines as B
from metrics import apk

K = B.K


def user_item_matrix(sp, users, items, value="sec"):
    """Dense-free helper: (rows, cols, vals) of history restricted to given users/items."""
    h = sp.hist.groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    uix = {u: i for i, u in enumerate(users)}
    iix = {t: j for j, t in enumerate(items)}
    h = h[h.user_id.isin(uix) & h.tv_show_id.isin(iix)]
    return h.user_id.map(uix).values, h.tv_show_id.map(iix).values, h[value].values.astype(np.float64)


def personal_scores(sp):
    """Phase A d_personal as a score matrix (sum of watched seconds over history, in pool)."""
    r, c, v = user_item_matrix(sp, sp.users, sp.pool)
    S = np.zeros((len(sp.users), len(sp.pool)))
    np.add.at(S, (r, c), v)
    return S, S > 0


def topk(S, valid, sp, fallback=None):
    """{user: 5 ids}: valid items by score desc (ties by id asc), then fallback."""
    fallback = fallback if fallback is not None else B.popularity(sp, in_pool=True)
    pool = np.asarray(sp.pool)
    M = np.where(valid, S, -np.inf)
    # candidates: top K by score; stable argsort on the K-partition keeps id order on ties
    part = np.argpartition(-M, K, axis=1)[:, :K + 1] if M.shape[1] > K + 1 else np.tile(np.arange(M.shape[1]), (len(M), 1))
    out = {}
    for row, u in enumerate(sp.users):
        cand = part[row]
        # include all items tied with the K-th best so tie-breaking by id is exact
        kth = np.sort(M[row, cand])[::-1][K - 1]
        if np.isfinite(kth):
            cand = np.union1d(cand, np.flatnonzero(M[row] == kth))
        cand = cand[np.isfinite(M[row, cand])]
        cand = cand[np.lexsort((cand, -M[row, cand]))][:K]
        out[u] = B.fill(pool[cand].tolist(), fallback)
    return out


def ap_per_user(target, preds):
    users = list(target)
    return users, np.array([apk(target[u], preds.get(u, []), K) for u in users])


def bootstrap_ci(ap, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ap), size=(n, len(ap)))
    m = ap[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def minmax_rows(S, valid):
    """Per-user min-max over valid pool items; invalid -> 0."""
    M = np.where(valid, S, np.nan)
    lo = np.nanmin(np.where(valid.any(1, keepdims=True), M, 0), axis=1, keepdims=True)
    hi = np.nanmax(np.where(valid.any(1, keepdims=True), M, 0), axis=1, keepdims=True)
    Z = (S - lo) / np.where(hi > lo, hi - lo, 1)
    return np.where(valid, Z, 0.0)


def rrf_rows(S, valid, k=60):
    """Reciprocal-rank contribution 1/(k + rank) per valid item (rank 1 = best; ties -> smaller id)."""
    M = np.where(valid, S, -np.inf)
    order = np.argsort(-M, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(M.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, M.shape[1] + 1)[None, :]
    return np.where(valid, 1.0 / (k + ranks), 0.0)
