"""Phase G: title families (schedule metadata only).

family_key(title): lower-case; keep a type prefix (т/с, х/ф, м/с, м/ф, д/ф, д/с, т/ф); strip quotes;
cut the title at the FIRST season / episode / series marker (", N сезон", "N эп.", "N с.", "N-M с.",
"серия N", ...), which removes the episode sub-title that follows it; drop remaining numbers; strip
punctuation; collapse spaces; if nothing but the prefix is left (e.g. "Х/ф '2012'"), keep the digits.
(A first-comma cut was tried first and merged different films such as "Я, робот" / "Я, Алекс Кросс"
and all numeric film titles, so the cut point is the first marker instead.)
Each tv_show_id is mapped through its MOST FREQUENT title over all schedules (ids carry episode titles).

Components:
- family personal: score(u, s) = user's history seconds summed over all ids in family(s);
- family rate: family seconds / family history airtime x s's target-window airtime;
- family fallback: pool ranked by family history seconds (all users), then id seconds, then id.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sps

import data

PREFIX = re.compile(r"^\s*([тхмдc]/[сфc])\s*\.?\s*")
MARKERS = re.compile(r"(\d+\s*(-\s*\d+)?\s*(сезон\w*|эп\w*\.?|с\.|сер\w*|вып\w*|ч\.|част\w*))|"
                     r"((сезон\w*|эп\w*\.?|сер\w*|вып\w*)\s*\d+)")


def family_key(title) -> str:
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return ""
    t = str(title).lower().replace("ё", "е")
    m = PREFIX.match(t)
    pre = m.group(1).replace("c", "с") + " " if m else ""
    t = t[m.end():] if m else t
    t = re.sub(r"[\"'«»“”„`]", "", t)
    m = MARKERS.search(t)
    if m:
        t = t[:m.start()]
    clean = lambda x: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", x)).strip()
    key = clean(re.sub(r"\d+", " ", t)) or clean(t)
    return (pre + key).strip()


def id_families(parts=("11-20", "21-30", "31-42")) -> pd.DataFrame:
    """tv_show_id -> title (most frequent), family, category, channel (most frequent)."""
    s = data.load_schedule_all(parts)
    s = s[s.tv_show_id != 0]
    top = (s.groupby(["tv_show_id", "tv_show_title"]).size().rename("n").reset_index()
           .sort_values(["tv_show_id", "n"], ascending=[True, False]).drop_duplicates("tv_show_id"))
    meta = s.groupby("tv_show_id").agg(category=("tv_show_category", "first"),
                                       n_channels=("channel_id", "nunique"))
    out = top.set_index("tv_show_id")[["tv_show_title"]].join(meta)
    out["family"] = out.tv_show_title.map(family_key)
    out.loc[out.family == "", "family"] = "id:" + out.index[out.family == ""].astype(str)
    return out


def _user_rows(users, uids):
    users = np.asarray(users, dtype=np.uint64)
    order = np.argsort(users)
    return order[np.searchsorted(users[order], np.asarray(uids, dtype=np.uint64))]


def family_scores(sp, fam: pd.Series, rate=False, hist_air=None, target_air=None):
    """fam: tv_show_id -> family. rate=True needs hist_air (id -> history airtime) and
    target_air (id -> target-window airtime)."""
    h = sp.hist[sp.hist.user_id.isin(sp.users)].groupby(["user_id", "tv_show_id"], as_index=False).sec.sum()
    h = h[h.tv_show_id.isin(fam.index)]
    fams = np.sort(fam.unique())
    fcol = np.searchsorted(fams, fam.reindex(h.tv_show_id).values)
    F = sps.csr_matrix((h.sec.values.astype(float), (_user_rows(sp.users, h.user_id.values), fcol)),
                       shape=(len(sp.users), len(fams)))
    F.sum_duplicates()
    pool = np.asarray(sp.pool)
    pf = fam.reindex(pool)
    known = pf.notna().values
    pcol = np.searchsorted(fams, pf.fillna(fams[0]).values)
    S = F[:, pcol].toarray()
    if rate:
        fa = hist_air.groupby(fam.reindex(hist_air.index).values).sum()
        denom = fa.reindex(pf.values).fillna(0).values
        mult = np.where(denom > 0, target_air.reindex(pool).fillna(0).values / np.maximum(denom, 1), 0)
        S = S * mult[None, :]
    S[:, ~known] = 0
    return S, S > 0


def family_fallback(sp, fam: pd.Series):
    h = sp.hist[sp.hist.tv_show_id.isin(sp.pool)]
    ids = h.groupby("tv_show_id").sec.sum()
    allh = sp.hist.groupby("tv_show_id").sec.sum()
    fsec = allh.groupby(fam.reindex(allh.index).values).sum()
    df = pd.DataFrame({"tv_show_id": np.asarray(sp.pool)})
    df["fsec"] = fsec.reindex(fam.reindex(df.tv_show_id).values).fillna(0).values
    df["isec"] = ids.reindex(df.tv_show_id).fillna(0).values
    return df.sort_values(["fsec", "isec", "tv_show_id"], ascending=[False, False, True]).tv_show_id.tolist()
