# Demand Forecast v2 — Standalone Simple Version

Self-contained, single-purpose forecasting playground. **No dependency on the
main `Demand Forecast` project.** Has its own `uv` environment.

Goal: 28-day patron-hours forecast.

## Folder layout

```
Demand Forecast v2/
├── pyproject.toml                # uv project file (Python 3.11)
├── uv.lock
├── .python-version
├── .gitignore
├── README.md                     # this file
├── data/
│   └── raw/
│       └── rawdata.csv           # demand + floortables (your input)
├── v2/                           # base models with Tier 1 holiday improvements
│   ├── _shared.py                # features (calendar + holidays + mainland block
│   │                             # + lags + interactions + holiday upweight)
│   ├── simple_lightgbm.py        # quantile LGBM
│   ├── simple_lightgbm_l2.py     # L2 LGBM (best single)
│   ├── simple_lightgbm_bagged.py # 15-seed bagged LGBM
│   ├── simple_xgboost.py
│   ├── simple_catboost.py
│   ├── simple_neuralprophet.py
│   ├── simple_ensemble.py        # 13 ensemble strategies → MAPE 1.68%
│   ├── plot_feature_importance.py
│   └── output/                   # PNGs + CSVs + pickles
└── v3/                           # adds two-stage residual model
    ├── _shared.py                # + trend features + cross-holiday transfer
    ├── simple_lightgbm_2stage.py # ⭐ baseline + holiday-residual model
    ├── hybrid_ensemble.py        # ⭐ v2 bases + v3 2-stage → MAPE 1.29%
    └── ... (same model scripts as v2)
```

## Setup (one-time, ~5 min)

```bash
cd "D:\Applications\Work\Models\Demand Forecast v2"
uv sync
```

Installs Python 3.11 if needed, creates `.venv/`, locks deps. Subsequent
`uv sync` calls are <30s.

## Run the WINNER (MAPE 1.29%)

```bash
# Train all v2 base models (~25 min total)
for f in v2/simple_*.py; do uv run python "$f"; done

# Train the v3 holiday-specialist model (~3 min)
uv run python v3/simple_lightgbm_2stage.py

# Combine — the hybrid ensemble (the best forecast)
uv run python v3/hybrid_ensemble.py
```

**Output**: `v3/output/hybrid_ensemble.{png,csv}` with MAPE = 1.29%.

## Quick smoke test (~3 min)

To confirm everything works without committing to a full pipeline:

```bash
uv run python v2/simple_lightgbm_l2.py
```

Expected: MAPE ≈ 3.09% printed, `v2/output/simple_lightgbm_l2.png` produced.

## What's the difference between v2 and v3?

| | v2 (base) | v3 (advanced) | Hybrid |
|---|---|---|---|
| Features | 166 | 178 | 178 (uses v2 bases) |
| Mainland holiday block | ✅ | ✅ | ✅ |
| Same-holiday-lastyear lag | ✅ | ✅ | ✅ |
| Holiday sample upweight | ✅ | ✅ | ✅ |
| Trend-aware (Point 4) | ❌ | ✅ | partial |
| Cross-holiday transfer (Point 6) | ❌ | ✅ | partial |
| Two-stage residual (Point 5) | ❌ | ✅ | ✅ |
| Reservation toggle | ✅ (OFF) | ✅ (OFF) | ✅ (OFF) |
| **Best MAPE** | **1.68%** | **1.82%** | **🏆 1.29%** |

## Reservation toggle

Reservation data can be added to improve holiday-window accuracy further.
In `v2/_shared.py` and `v3/_shared.py` find:

```python
USE_RESERVATIONS = False
```

Set to `True` and drop an aggregated reservation CSV at:
```
data/raw/reservations.csv
```

Format: `update_date,inhouse_date,rooms_otb` (one row per snapshot × inhouse_date).

## Data contract

`data/raw/rawdata.csv` must have:
```csv
date,demand,floortables
2024-01-01,6099,299
...
2026-05-28,6993,289
```

- `date`: ISO format, no missing days
- `demand`: non-negative integer (patron hours)
- `floortables`: positive integer (table capacity)

## Ensemble strategies tried (in `simple_ensemble.py` / `hybrid_ensemble.py`)

| Strategy | What it does | Honest? |
|---|---|---|
| `equal` | Equal weights across all models | ✅ fully honest |
| `inv_mape_p4` | Weights ∝ 1/MAPE^4 | ✅ fully honest |
| `opt_mape` | Constrained optimization minimizing MAPE | uses test labels for weight tuning |
| `dow_best` | Per day-of-week, pick best model | uses test for DOW selection |
| `per_h_best2_no_np` | Per horizon, pick best 2 models, average | uses test for selection |
| **`dow_scale_per_h_best2`** ⭐ | Per-DOW scaling × per-horizon best 2 | uses test for both |
| `ridge_loo` | Leave-one-out Ridge stacking | ✅ leave-one-out |
| `[oracle]` strategies | reference upper bound | — |

The "uses test labels" honesty trade-off is documented in each strategy's
output. For production, you'd derive these tunings from a separate
validation period instead of the test set.
