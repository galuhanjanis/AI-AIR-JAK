
from pathlib import Path
import pandas as pd
import numpy as np
from .transformer_numpy import FEATURES, POLLUTANTS

def load_data(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

def station_rows(df, station):
    return df[df["Station_Name"] == station].sort_values("datetime").copy()

def feature_row(row):
    X = pd.DataFrame([{f: pd.to_numeric(row.get(f), errors="coerce") for f in FEATURES}])
    missing = X.columns[X.isna().any()].tolist()
    if missing:
        raise ValueError("Missing feature(s): " + ", ".join(missing))
    return X

def historical_quantiles(df):
    q = {}
    for p in POLLUTANTS:
        s = pd.to_numeric(df[p], errors="coerce").dropna()
        q[p] = {
            "q50": float(s.quantile(.50)),
            "q75": float(s.quantile(.75)),
            "q90": float(s.quantile(.90)),
        }
    return q

def relative_risk(v, q):
    if v >= q["q90"]: return "Very High"
    if v >= q["q75"]: return "High"
    if v >= q["q50"]: return "Moderate"
    return "Low"
