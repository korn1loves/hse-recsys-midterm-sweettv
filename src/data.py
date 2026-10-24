"""Loading + parsing of the sweet.tv data with explicit timestamp formats and parquet caching."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sweettv-tv-program-recommender"
CACHE = ROOT / "results" / "cache"

LOG_FMT = "%Y-%m-%d %H:%M:%S"     # dataset11-30.csv
SCHED_FMT = "%d.%m.%Y %H:%M:%S"   # export_arh_*.csv (day first)

SCHED_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}


def iso_week(ts: pd.Series) -> pd.Series:
    return ts.dt.isocalendar().week.astype("int16")


def load_log() -> pd.DataFrame:
    p = CACHE / "log.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_csv(DATA / "dataset11-30.csv",
                     dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"})
    df["start_time"] = pd.to_datetime(df["start_time"], format=LOG_FMT)
    df["stop_time"] = pd.to_datetime(df["stop_time"], format=LOG_FMT)
    df["week"] = iso_week(df["start_time"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def load_schedule(part: str) -> pd.DataFrame:
    p = CACHE / f"sched_{part}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = pd.read_csv(DATA / SCHED_FILES[part], dtype={"tv_show_id": "int64", "channel_id": "int32",
                                                       "duration": "int64"}, low_memory=False)
    for c in ["year_of_production", "director", "actors", "tv_show_genre_1", "tv_show_genre_2",
              "tv_show_genre_3", "tv_show_category", "tv_show_title", "channel_title"]:
        df[c] = df[c].astype("string")
    df["start_time"] = pd.to_datetime(df["start_time"], format=SCHED_FMT)
    df["end_time"] = df["start_time"] + pd.to_timedelta(df["duration"], unit="s")
    df["week"] = iso_week(df["start_time"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return df


def load_schedule_all(parts=("11-20", "21-30")) -> pd.DataFrame:
    return pd.concat([load_schedule(x) for x in parts], ignore_index=True)


def load_submission() -> pd.DataFrame:
    return pd.read_csv(DATA / "submission.csv", dtype={"user_id": "uint64", "tv_show_id": "string"})
