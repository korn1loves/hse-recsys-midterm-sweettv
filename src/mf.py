"""Method 1: implicit-feedback ALS (Hu, Koren, Volinsky 2008) in numpy/scipy.

Matrix: ALL log users x labelled shows seen in history (+ pool shows, which get zero factors if
never watched). Preference p_ui = 1 if the user has any history seconds on the show;
confidence c_ui = 1 + alpha * log1p(seconds / 60). Loss: sum c_ui (p_ui - x_u.y_i)^2 + reg(|X|^2+|Y|^2).
"""
import time

import numpy as np
import scipy.sparse as sps


def _solve_side(Cm1, F, reg):
    """One ALS half-step. Cm1: csr (n x m) with entries c-1 (>0 where p=1); F: (m x k) fixed factors.
    x_u = (F'F + F_u'(C_u - I)F_u + reg I)^-1 F_u' C_u p_u."""
    k = F.shape[1]
    FtF = F.T @ F + reg * np.eye(k)
    out = np.zeros((Cm1.shape[0], k))
    indptr, indices, data = Cm1.indptr, Cm1.indices, Cm1.data
    for u in range(Cm1.shape[0]):
        a, b = indptr[u], indptr[u + 1]
        if a == b:
            continue
        Fu = F[indices[a:b]]
        cm1 = data[a:b]
        A = FtF + (Fu.T * cm1) @ Fu
        rhs = Fu.T @ (cm1 + 1.0)
        out[u] = np.linalg.solve(A, rhs)
    return out


class ALS:
    def __init__(self, factors=64, reg=0.1, alpha=10.0, iterations=15, seed=0):
        self.factors, self.reg, self.alpha, self.iterations, self.seed = factors, reg, alpha, iterations, seed

    def fit(self, rows, cols, secs, n_users, n_items):
        t0 = time.time()
        conf = self.alpha * np.log1p(secs / 60.0)
        Cui = sps.csr_matrix((conf, (rows, cols)), shape=(n_users, n_items))
        Cui.sum_duplicates()
        Cui.data = np.maximum(Cui.data, 1e-9)  # keep zero-second rows as observed positives
        Ciu = Cui.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.U = rng.normal(0, 0.01, (n_users, self.factors))
        self.V = rng.normal(0, 0.01, (n_items, self.factors))
        for _ in range(self.iterations):
            self.U = _solve_side(Cui, self.V, self.reg)
            self.V = _solve_side(Ciu, self.U, self.reg)
        self.fit_time = time.time() - t0
        return self


def mf_scores(sp, factors=64, reg=0.1, alpha=10.0, iterations=15, seed=0):
    """Fit on sp.hist (all users) and score sp.users x sp.pool. Users without history -> invalid."""
    h = sp.hist.groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    users = np.union1d(h.user_id.unique(), np.asarray(sp.users, dtype=h.user_id.dtype))
    items = np.union1d(h.tv_show_id.unique(), sp.pool)
    # positional lookup via searchsorted (dict/Series.map on uint64 ids is unreliable)
    uix = lambda x: np.searchsorted(users, np.asarray(x, dtype=users.dtype))
    iix = lambda x: np.searchsorted(items, np.asarray(x, dtype=items.dtype))
    m = ALS(factors, reg, alpha, iterations, seed).fit(
        uix(h.user_id.values), iix(h.tv_show_id.values), h.sec.values.astype(float),
        len(users), len(items))
    t0 = time.time()
    Uu = m.U[uix(sp.users)]
    Vp = m.V[iix(sp.pool)]
    S = Uu @ Vp.T
    has_hist = np.isin(np.asarray(sp.users), h.user_id.unique())
    valid = np.repeat(has_hist[:, None], len(sp.pool), axis=1)
    return S, valid, {"fit_s": m.fit_time, "predict_s": time.time() - t0}
