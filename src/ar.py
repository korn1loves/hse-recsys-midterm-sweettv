"""Method 3: item-to-item association rules a -> b over users' watched-show sets.

Transaction = set of shows a user watched for >= SET_MIN_SEC seconds in total over history
(all log users). Rule a -> b kept if support(a,b) = n_ab / n_users >= MIN_SUPPORT and
confidence = n_ab / n_a >= MIN_CONF (a != b). Score(u, b) = sum of confidences of kept rules
a -> b over a in the user's set. Valid items: score > 0.
"""
import time

import numpy as np
import scipy.sparse as sps

SET_MIN_SEC = 300
MIN_SUPPORT = 0.01
MIN_CONF = 0.1


def ar_scores(sp, set_min_sec=SET_MIN_SEC, min_support=MIN_SUPPORT, min_conf=MIN_CONF):
    t0 = time.time()
    h = sp.hist.groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    h = h[h.sec >= set_min_sec]
    users = np.union1d(h.user_id.unique(), np.asarray(sp.users, dtype=h.user_id.dtype))
    items = np.union1d(h.tv_show_id.unique(), sp.pool)
    # positional lookup via searchsorted (dict/Series.map on uint64 ids is unreliable)
    uix = lambda x: np.searchsorted(users, np.asarray(x, dtype=users.dtype))
    iix = lambda x: np.searchsorted(items, np.asarray(x, dtype=items.dtype))
    Bm = sps.csr_matrix((np.ones(len(h)), (uix(h.user_id.values), iix(h.tv_show_id.values))),
                        shape=(len(users), len(items)))
    n_users_active = len(h.user_id.unique())
    co = (Bm.T @ Bm).tocoo()
    cnt = np.asarray(Bm.sum(0)).ravel()
    keep = (co.row != co.col) & (co.data / n_users_active >= min_support)
    conf = co.data[keep] / cnt[co.row[keep]]
    k2 = conf >= min_conf
    R = sps.csr_matrix((conf[k2], (co.row[keep][k2], co.col[keep][k2])), shape=(len(items),) * 2)
    fit = time.time() - t0
    t1 = time.time()
    Bu = Bm[uix(sp.users)]
    S = (Bu @ R[:, iix(sp.pool)]).toarray()
    return S, S > 0, {"fit_s": fit, "predict_s": time.time() - t1, "n_rules": int(R.nnz),
                      "n_transactions": n_users_active}
