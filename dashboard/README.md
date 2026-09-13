# XLinear Weather Forecasting Demo

## Purpose
This application is a local academic demonstration of the official implementation of **XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs** (AAAI 2026). It demonstrates interactive inference on held-out test windows from the Weather benchmark using the locally reproduced model weights (`features='MS'`).

## Requirements
* **Python:** 3.11.x
* **Virtual Environment:** Existing repository `.venv`
* **Dataset:** `dataset/weather.csv`
* **Checkpoint:** Trained XLinear MS checkpoint under `checkpoints/weather_96_96_XLinear_custom_ftMS_sl96_ll48_pl96_dm1024_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth`
* **Dependencies:** `streamlit`, `torch==2.6.0+cpu`, `numpy==1.26.4`, `pandas==2.1.3`, `scikit-learn==1.6.1`, `matplotlib`

## Run
From the repository root with the active virtual environment:

```powershell
python -m streamlit run dashboard/app.py
```

## Demo Flow
1. **Select held-out test window:** Adjust the sidebar slider to pick any sample index between 0 and 10,431 from the test split.
2. **View historical OT:** Inspect the 96 historical timesteps of the target variable `OT` alongside 20 exogenous series.
3. **Run XLinear inference:** The model executes forward propagation on CPU in single-digit milliseconds.
4. **View predicted and actual future OT:** Observe the 96-step forecast plotted continuously against ground-truth future observations.
5. **Inspect per-window errors:** Compare window-specific Scaled MSE, Scaled MAE, and Original-scale MAE metrics.

## Limitations
* **Academic benchmark demonstration:** Designed strictly for retrospective evaluation on historical test splits, not a live or operational weather forecasting service.
* **Fixed dataset & task:** Configured specifically for the Weather dataset with a 96-step lookback and 96-step forecast horizon predicting `OT`.
* **Deterministic point forecasts:** Produces point predictions without calibrated empirical uncertainty or confidence intervals.
* **No online ingestion:** Uses static local benchmark CSV data without live sensor feeds or online retraining.
* **Hardware context:** Inference runs on local CPU.
