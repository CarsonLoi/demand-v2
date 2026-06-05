"""Simplified NeuralProphet — single-file end-to-end test.

Trend + weekly + yearly + auto-regression (n_lags=28). No holidays / regressors
(NeuralProphet v0.9 has a known bug combining n_lags > 0 with future regressors).

Run:
  uv run python v1/simple_neuralprophet.py
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.getLogger("NP").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from neuralprophet import NeuralProphet, set_log_level
set_log_level("ERROR")

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "raw" / "rawdata.csv"
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUT_DAYS = 28


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=== 1/4  Loading data ===")
df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
print(f"  {len(df)} rows  ({df.date.min().date()} .. {df.date.max().date()})")

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
print("\n=== 2/4  Splitting train/test ===")
train = df.iloc[:-HOLDOUT_DAYS].copy()
test = df.iloc[-HOLDOUT_DAYS:].copy()
print(f"  train: {len(train)} rows  ({train.date.min().date()} .. {train.date.max().date()})")
print(f"  test:  {len(test)} rows  ({test.date.min().date()}  .. {test.date.max().date()})")

# ---------------------------------------------------------------------------
# 3. Fit NeuralProphet
# ---------------------------------------------------------------------------
print("\n=== 3/4  Fitting NeuralProphet (this takes ~30-60s)... ===")
np_train = train.rename(columns={"date": "ds", "demand": "y"})[["ds", "y"]]
model = NeuralProphet(
    n_lags=28,
    n_forecasts=HOLDOUT_DAYS,   # forecast exactly the holdout length
    yearly_seasonality=4,
    weekly_seasonality=True,
    daily_seasonality=False,
    quantiles=[0.1, 0.5, 0.9],
    epochs=50,
    batch_size=64,
    learning_rate=0.01,
    trend_reg=0.5,
    ar_reg=0.05,
    seasonality_reg=0.1,
    growth="linear",
    loss_func="Huber",
)
model.fit(np_train, freq="D", progress=None, minimal=True)
print("  fitted")

# ---------------------------------------------------------------------------
# 4. Predict, compute metrics, plot
# ---------------------------------------------------------------------------
print("\n=== 4/4  Predicting + plotting ===")
future = model.make_future_dataframe(df=np_train, n_historic_predictions=False,
                                     periods=HOLDOUT_DAYS)
fc = model.predict(future)

# NeuralProphet outputs yhat1, yhat2, ... yhatN (predictions made 1..N steps ahead).
# For each test date i (1..28), pick yhat<i> from the row where ds == test_date_i.
y_p10, y_p50, y_p90 = [], [], []
train_end = train["date"].max()
for i, test_date in enumerate(test["date"], start=1):
    row = fc[fc["ds"] == test_date]
    if row.empty:
        y_p10.append(np.nan); y_p50.append(np.nan); y_p90.append(np.nan)
        continue
    col_50 = f"yhat{i}"
    col_10 = f"yhat{i} 10.0%"
    col_90 = f"yhat{i} 90.0%"
    y_p50.append(max(0, float(row[col_50].iloc[0])))
    y_p10.append(max(0, float(row[col_10].iloc[0])) if col_10 in row.columns else np.nan)
    y_p90.append(max(0, float(row[col_90].iloc[0])) if col_90 in row.columns else np.nan)

y_true = test["demand"].to_numpy(dtype=float)
y_p10 = np.array(y_p10)
y_p50 = np.array(y_p50)
y_p90 = np.array(y_p90)

mask = ~np.isnan(y_p50)
wape = np.abs(y_true[mask] - y_p50[mask]).sum() / np.abs(y_true[mask]).sum()
mape = np.mean(np.abs((y_true[mask] - y_p50[mask]) / y_true[mask]))
rmse = np.sqrt(np.mean((y_true[mask] - y_p50[mask]) ** 2))
bias = float(np.mean(y_p50[mask] - y_true[mask]))
cov_mask = mask & ~np.isnan(y_p10) & ~np.isnan(y_p90)
in_band = float(np.mean((y_true[cov_mask] >= y_p10[cov_mask])
                        & (y_true[cov_mask] <= y_p90[cov_mask])))

print(f"  WAPE          {wape*100:5.2f}%")
print(f"  MAPE          {mape*100:5.2f}%")
print(f"  RMSE          {rmse:8.0f}")
print(f"  Bias          {bias:+8.0f}")
print(f"  P80 coverage  {in_band*100:5.1f}%  (target 80%)")

# Plot
recent_train = train.tail(60)
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(recent_train["date"], recent_train["demand"], "o-", color="black",
        markersize=3, alpha=0.6, label="actual (train, trailing 60d)")
ax.plot(test["date"], y_true, "o-", color="black", markersize=5,
        label="actual (test, 28d holdout)")
ax.plot(test["date"], y_p50, "s-", color="C2", markersize=4,
        label="NeuralProphet P50 forecast")
ax.fill_between(test["date"], y_p10, y_p90, color="C2", alpha=0.20,
                label="P10-P90 interval")
ax.axvline(train_end, color="red", linestyle="--", alpha=0.5,
           label=f"train/test split ({train_end.date()})")
ax.set_xlabel("date")
ax.set_ylabel("demand (patron hours)")
ax.set_title(f"Simple NeuralProphet | WAPE={wape*100:.2f}%  MAPE={mape*100:.2f}%  "
             f"coverage={in_band*100:.0f}%")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
plt.xticks(rotation=30)
plt.tight_layout()

out_path = OUT / "simple_neuralprophet.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  -> {out_path}")

result = pd.DataFrame({
    "date": test["date"].values,
    "actual": y_true,
    "p10": y_p10,
    "p50": y_p50,
    "p90": y_p90,
    "abs_error": np.abs(y_true - y_p50),
    "pct_error": (y_true - y_p50) / y_true,
})
csv_path = OUT / "simple_neuralprophet.csv"
result.to_csv(csv_path, index=False)
print(f"  -> {csv_path}")

# Save preds.pkl for ensemble use
import pickle as _pkl
preds = {
    "target_date": list(test["date"].values),
    "y_true": list(y_true),
    "y_p10": list(y_p10),
    "y_p50": list(y_p50),
    "y_p90": list(y_p90),
}
with open(OUT / "simple_neuralprophet_preds.pkl", "wb") as f:
    _pkl.dump(preds, f)
print(f"  -> {OUT / 'simple_neuralprophet_preds.pkl'}")
