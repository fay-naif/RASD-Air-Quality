from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "air_quality.csv"
PREDICTORS = [
    "PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM",
    "pm25_lag_1","pm25_lag_3","pm25_lag_6","pm25_lag_24",
    "pm25_past_3h_mean","pm25_past_6h_mean","pm25_past_24h_mean",
    "hour_sin","hour_cos","month_sin","month_cos",
]
CLASS_NAMES = ["منخفض","مرتفع قليلاً","مرتفع","مرتفع جداً"]

def load_real_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Missing data/air_quality.csv")
    return pd.read_csv(DATA_FILE)

def classify_pm25(value):
    value=float(value)
    if value<=12: return "منخفض"
    if value<=35.4: return "مرتفع قليلاً"
    if value<=55.4: return "مرتفع"
    return "مرتفع جداً"

def prepare_data(raw):
    d=raw.copy()
    d["timestamp"]=pd.to_datetime(d[["year","month","day","hour"]])
    d=d.sort_values("timestamp").drop_duplicates("timestamp").copy()
    numeric=["PM2.5","PM10","SO2","NO2","CO","O3","TEMP","PRES","DEWP","RAIN","WSPM"]
    d[numeric]=d[numeric].apply(pd.to_numeric,errors="coerce").ffill(limit=6)
    d["hour_sin"]=np.sin(2*np.pi*d["timestamp"].dt.hour/24)
    d["hour_cos"]=np.cos(2*np.pi*d["timestamp"].dt.hour/24)
    d["month_sin"]=np.sin(2*np.pi*d["timestamp"].dt.month/12)
    d["month_cos"]=np.cos(2*np.pi*d["timestamp"].dt.month/12)
    for lag in (1,3,6,24): d[f"pm25_lag_{lag}"]=d["PM2.5"].shift(lag)
    for window in (3,6,24):
        d[f"pm25_past_{window}h_mean"]=d["PM2.5"].shift(1).rolling(window,min_periods=window).mean()
    next_time=d["timestamp"].shift(-1)
    d["target_pm25"]=d["PM2.5"].shift(-1)
    d.loc[(next_time-d["timestamp"])!=pd.Timedelta(hours=1),"target_pm25"]=np.nan
    d["target_class"]=d["target_pm25"].map(classify_pm25)
    return d.dropna(subset=PREDICTORS+["target_pm25","target_class"]).reset_index(drop=True)
