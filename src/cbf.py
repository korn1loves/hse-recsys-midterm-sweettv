"""Method 2: content-based filtering on schedule attributes.

Item vector = TF-IDF (L2-normalised) over prefixed tokens: cat:<category>, g:<genre 1..3>,
dir:<director names>, act:<actor names>, yr:<decade of year_of_production>; min_df=2.
Item attributes are taken from the first non-null value per show over the schedules passed
(training schedules + the target-window schedule; no viewing data involved).
User profile = log1p(seconds)-weighted mean of the user's history item vectors; score = cosine
between profile and pool item vectors. Valid items: score > 0.
"""
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import data

ATTRS = ["tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
         "director", "actors", "year_of_production"]


def item_table(parts) -> pd.DataFrame:
    s = data.load_schedule_all(parts)
    s = s[s.tv_show_id != 0]
    return s.groupby("tv_show_id")[ATTRS + ["tv_show_title"]].first()


def _names(x):
    return [] if pd.isna(x) else [n.strip().lower() for n in str(x).split(",") if n.strip()]


def tokens(row) -> list:
    t = []
    if pd.notna(row.tv_show_category):
        t.append("cat:" + row.tv_show_category.lower())
    for g in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        if pd.notna(row[g]):
            t.append("g:" + row[g].lower())
    t += ["dir:" + n for n in _names(row.director)]
    t += ["act:" + n for n in _names(row.actors)]
    m = re.match(r"(\d{4})", str(row.year_of_production)) if pd.notna(row.year_of_production) else None
    if m:
        t.append(f"yr:{int(m.group(1)) // 10 * 10}s")
    return t


def item_vectors(items: pd.DataFrame):
    vec = TfidfVectorizer(analyzer=lambda x: x, min_df=2, norm="l2", sublinear_tf=False)
    X = vec.fit_transform([tokens(r) for _, r in items.iterrows()])
    return X.tocsr(), vec


def cbf_scores(sp, parts):
    t0 = time.time()
    it = item_table(parts)
    X, vec = item_vectors(it)
    iix = {t: j for j, t in enumerate(it.index)}
    h = sp.hist[sp.hist.user_id.isin(sp.users)].groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    h = h[h.tv_show_id.isin(iix)]
    uix = {u: i for i, u in enumerate(sp.users)}
    W = sps.csr_matrix((np.log1p(h.sec.values.astype(float)),
                        (h.user_id.map(uix).values, h.tv_show_id.map(iix).values)),
                       shape=(len(sp.users), len(it)))
    P = normalize(W @ X, norm="l2")  # mean then L2 = same direction as weighted sum
    fit = time.time() - t0
    t1 = time.time()
    pool_rows = np.array([iix.get(t, -1) for t in sp.pool])
    Xp = X[np.clip(pool_rows, 0, None)]
    S = (P @ Xp.T).toarray()
    S[:, pool_rows < 0] = 0
    return S, S > 0, {"fit_s": fit, "predict_s": time.time() - t1, "n_features": X.shape[1],
                      "n_items": X.shape[0]}
