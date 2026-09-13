import os
import sys
import time
import json
from types import SimpleNamespace

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from data_provider.data_factory import data_provider
from utils.metrics import metric

def main():
    start_time = time.time()

    # Step 2: Configuration matching XLinear Weather 96-96 experiment
    args = SimpleNamespace(
        data='custom',
        root_path='./dataset/',
        data_path='weather.csv',
        features='MS',
        target='OT',
        seq_len=96,
        label_len=48,
        pred_len=96,
        batch_size=32,
        num_workers=0,
        embed='timeF',
        freq='h'
    )

    print(">>> Instantiating official test data loader...")
    test_data, test_loader = data_provider(args, flag='test')
    test_data_len = len(test_data)
    print(f"Test dataset total samples: {test_data_len}")

    preds_list = []
    trues_list = []
    first_batch_sample_0_last_ot = None

    # Step 3: Iterate batches and generate persistence forecast
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        # batch_x: [B, 96, 21], batch_y: [B, 144, 21]
        # Last feature column is OT
        last_ot = batch_x[:, -1:, -1:]  # [B, 1, 1]
        pred = last_ot.repeat(1, args.pred_len, 1)  # [B, 96, 1]
        true = batch_y[:, -args.pred_len:, -1:]  # [B, 96, 1]

        if i == 0:
            first_batch_sample_0_last_ot = float(batch_x[0, -1, -1].item())

        preds_list.append(pred.detach().cpu().numpy())
        trues_list.append(true.detach().cpu().numpy())

    # Step 4: Concatenate and verify
    preds = np.concatenate(preds_list, axis=0)  # [10432, 96, 1]
    trues = np.concatenate(trues_list, axis=0)  # [10432, 96, 1]

    evaluated_windows = preds.shape[0]
    expected_windows = 10432
    print(f"Evaluated windows: {evaluated_windows} (Expected: {expected_windows})")
    assert evaluated_windows == expected_windows, f"Window count mismatch: {evaluated_windows} != {expected_windows}"
    assert preds.shape == (expected_windows, 96, 1), f"Preds shape mismatch: {preds.shape}"
    assert trues.shape == (expected_windows, 96, 1), f"Trues shape mismatch: {trues.shape}"

    assert np.all(np.isfinite(preds)), "Non-finite values found in predictions!"
    assert np.all(np.isfinite(trues)), "Non-finite values found in ground truth!"
    assert np.isnan(preds).sum() == 0, "NaNs found in predictions!"
    assert np.isnan(trues).sum() == 0, "NaNs found in ground truth!"
    assert np.isinf(preds).sum() == 0, "Infs found in predictions!"
    assert np.isinf(trues).sum() == 0, "Infs found in ground truth!"

    # Step 8: Sanity assertions
    assert np.allclose(preds[0, :, 0], preds[0, 0, 0]), "Persistence prediction is not constant across horizon!"
    assert np.isclose(preds[0, 0, 0], first_batch_sample_0_last_ot, atol=1e-6), \
        f"Forecast {preds[0, 0, 0]} does not match input history OT {first_batch_sample_0_last_ot}!"
    print("Sanity checks passed: Prediction is strictly constant persistence of last observed OT.")

    # Step 5: Metrics calculation
    mae, mse, rmse, mape, mspe, rse, corr, nse, kge, r2 = metric(preds, trues)
    corr_scalar = float(np.mean(corr)) if isinstance(corr, np.ndarray) else float(corr)

    print("\n" + "="*50)
    print("PERSISTENCE BASELINE — WEATHER 9696")
    print("="*50)
    print(f"MSE:  {float(mse):.7f}")
    print(f"MAE:  {float(mae):.7f}")
    print(f"RMSE: {float(rmse):.7f}")
    print(f"MAPE: {float(mape):.7f}")
    print(f"MSPE: {float(mspe):.7f}")
    print(f"RSE:  {float(rse):.7f}")
    print(f"CORR: {corr_scalar:.7f}")
    print(f"NSE:  {float(nse):.7f}")
    print(f"KGE:  {float(kge):.7f}")
    print(f"R2:   {float(r2):.7f}")

    # Step 6: Comparison against XLinear
    xlinear_mse = 0.001319668721407652
    xlinear_mae = 0.026476873084902763

    mse_abs_diff = float(mse) - xlinear_mse
    mse_rel_pct = (mse_abs_diff / float(mse)) * 100.0

    mae_abs_diff = float(mae) - xlinear_mae
    mae_rel_pct = (mae_abs_diff / float(mae)) * 100.0

    print("\n" + "-"*50)
    print("COMPARISON: XLinear vs Persistence Baseline")
    print("-"*50)
    print(f"Persistence MSE:               {float(mse):.7f}")
    print(f"XLinear MSE:                   {xlinear_mse:.7f}")
    print(f"MSE Absolute Improvement:      {mse_abs_diff:.7f}")
    print(f"MSE Relative Improvement (%):  {mse_rel_pct:.4f}%")
    print(f"Persistence MAE:               {float(mae):.7f}")
    print(f"XLinear MAE:                   {xlinear_mae:.7f}")
    print(f"MAE Absolute Improvement:      {mae_abs_diff:.7f}")
    print(f"MAE Relative Improvement (%):  {mae_rel_pct:.4f}%")

    # Step 7: Save reproducible artifacts
    os.makedirs('baseline_results', exist_ok=True)

    pred_path = os.path.join('baseline_results', 'weather_persistence_96_predictions.npy')
    true_path = os.path.join('baseline_results', 'weather_persistence_96_truth.npy')
    json_path = os.path.join('baseline_results', 'weather_persistence_96_metrics.json')

    np.save(pred_path, preds)
    np.save(true_path, trues)

    metrics_payload = {
        "baseline": "Persistence",
        "dataset": "Weather",
        "target": "OT",
        "features": "MS",
        "seq_len": 96,
        "pred_len": 96,
        "test_windows": int(evaluated_windows),
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "rse": float(rse),
        "corr": float(corr_scalar),
        "nse": float(nse),
        "kge": float(kge),
        "r2": float(r2),
        "xlinear_local_mse": float(xlinear_mse),
        "xlinear_local_mae": float(xlinear_mae),
        "mse_relative_improvement_percent": float(mse_rel_pct),
        "mae_relative_improvement_percent": float(mae_rel_pct)
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nSaved artifacts to baseline_results/")
    print(f"Total runtime: {elapsed:.2f} seconds")

if __name__ == '__main__':
    main()
