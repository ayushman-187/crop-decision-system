"""data.py - constants, dataset loading and input handling.

Everything that is "about the data" lives here so the other modules stay free of file paths and magic numbers.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
PROFIT_DEMO_PATH = ROOT / "data" / "profit_index_ILLUSTRATIVE_DEMO.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
SOIL_TEST_FEATURES = ["N", "P", "K"]            # need a lab/soil-test kit
CONTROLLABLE = ["N", "P", "K", "ph"]            # a farmer can change these (fertiliser, lime/sulphur)

LABEL = dict(N="nitrogen", P="phosphorus", K="potassium", temperature="temperature",
             humidity="humidity", ph="soil pH", rainfall="rainfall")
UNITS = dict(N="", P="", K="", temperature="°C", humidity="%", ph="", rainfall="mm")

# slider config used by the app: feature -> (min, max, default, step)
SLIDERS = dict(N=(0, 140, 50, 1), P=(5, 145, 50, 1), K=(5, 205, 40, 1),
               temperature=(8.0, 44.0, 25.0, 0.1), humidity=(14.0, 100.0, 70.0, 0.5),
               ph=(3.5, 9.9, 6.5, 0.1), rainfall=(20.0, 300.0, 100.0, 1.0))

# coarse agronomic groups, used ONLY for the fairness / per-group analysis in evaluate.py
CROP_GROUP = {
    "rice": "cereal", "maize": "cereal",
    "jute": "fibre", "cotton": "fibre",
    "chickpea": "pulse", "kidneybeans": "pulse", "pigeonpeas": "pulse", "mothbeans": "pulse",
    "mungbean": "pulse", "blackgram": "pulse", "lentil": "pulse",
    "pomegranate": "fruit", "banana": "fruit", "mango": "fruit", "grapes": "fruit", "watermelon": "fruit",
    "muskmelon": "fruit", "apple": "fruit", "orange": "fruit", "papaya": "fruit",
    "coconut": "plantation", "coffee": "plantation",
}

# ready-made example fields (A/B/C are the demo fields used in the report)
EXAMPLE_FIELDS = {
    "A - moderate field (jute vs rice)": dict(N=85, P=45, K=40, temperature=24.0, humidity=82.0, ph=6.3, rainfall=190.0),
    "B - borderline field (pulses)": dict(N=32.4, P=58.0, K=19.5, temperature=29.1, humidity=74.4, ph=7.0, rainfall=59.6),
    "C - rice, low N, acidic": dict(N=45, P=47, K=40, temperature=23.7, humidity=82.2, ph=5.0, rainfall=233.1),
}


def load_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the Kaggle 'Crop Recommendation' CSV and sanity-check it."""
    df = pd.read_csv(path or DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    return df


def load_profit_index(path: Path | str | None = None) -> dict:
    """crop -> profit index in [0,1]. Blank cells are skipped. The file is NOT real market data unless YOU fill it."""
    p = Path(path or PROFIT_DEMO_PATH)
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d.dropna(subset=["profit_index_0_to_1"])
    return dict(zip(d["crop"], d["profit_index_0_to_1"].astype(float)))


def clip_physical(X: pd.DataFrame) -> pd.DataFrame:
    """Keep simulated / user values physically valid (no negative nutrients, pH in 3-10, humidity <= 100)."""
    X = X.copy()
    X["ph"] = X["ph"].clip(3, 10)
    for c in ["N", "P", "K", "humidity", "rainfall"]:
        X[c] = X[c].clip(lower=0)
    X["humidity"] = X["humidity"].clip(upper=100)
    return X


def perturb_inputs(X: pd.DataFrame, std: pd.Series, sigma: float, n: int, rng) -> pd.DataFrame:
    """n noisy copies of every row. Noise = N(0,1) * sigma * feature std (cheap-sensor / seasonal error).
    NaN (= 'not provided') stays NaN."""
    reps = pd.concat([X[FEATURES]] * n, ignore_index=True)
    noisy = reps + rng.normal(0, 1, reps.shape) * sigma * std.values
    return clip_physical(noisy)


def field_series(d: dict) -> pd.Series:
    """dict -> float Series in canonical feature order; missing keys / None become NaN ('not provided')."""
    return pd.Series({f: (np.nan if d.get(f) is None else float(d[f])) for f in FEATURES})
