import os
import sys
import time
import json
import random
from types import SimpleNamespace

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from utils.metrics import metric

# ---------------------------------------------------------------------
# DLinear Architecture Implementation
# ---------------------------------------------------------------------
class moving_avg(nn.Module):
    """
    Moving average block to extract trend with symmetric edge-padding.
    """
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_len, Channel]
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)  # [Batch, Seq_len + 2*pad_len, Channel]
        x = self.avg(x.permute(0, 2, 1))      # [Batch, Channel, Seq_len]
        return x.permute(0, 2, 1)             # [Batch, Seq_len, Channel]


class series_decomp(nn.Module):
    """
    Series decomposition into seasonal/remainder and trend components.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Faithful minimal DLinear architecture (individual=False).
    Decomposition + two linear projections.
    """
    def __init__(self, seq_len=96, pred_len=96, moving_avg_kernel=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomposition = series_decomp(moving_avg_kernel)

        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # Initialize linear weights to 1 / seq_len matching reference DLinear
        self.Linear_Seasonal.weight = nn.Parameter(
            (1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
        )

    def forward(self, x):
        # x: [Batch, Seq_len, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        # Permute to [Batch, Channel, Seq_len] for temporal projection
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        output = seasonal_output + trend_output
        return output.permute(0, 2, 1)  # [Batch, Pred_len, Channel]


# ---------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------
def main():
    total_start_time = time.time()

    # Step 5: Reproducibility controls
    SEED = 2025
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Random seed: {SEED}")

    # Step 3: Dataset pipeline configuration matching XLinear Weather 96-96
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

    print(">>> Loading Weather dataset via official data_provider...")
    train_data, train_loader = data_provider(args, flag='train')
    vali_data, vali_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

    print(f"Train dataset length: {len(train_data)}")
    print(f"Validation dataset length: {len(vali_data)}")
    print(f"Test dataset length: {len(test_data)}")

    # Step 2: Instantiate DLinear model
    model = DLinear(seq_len=args.seq_len, pred_len=args.pred_len, moving_avg_kernel=25).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DLinear Total Parameters: {total_params}")
    print(f"DLinear Trainable Parameters: {trainable_params}")

    # Step 6: Optimizer, Criterion, Early Stopping setup
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 20
    PATIENCE = 3

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    os.makedirs('baseline_results', exist_ok=True)
    best_ckpt_path = os.path.join('baseline_results', 'dlinear_weather_96_checkpoint.pth')

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    history = []

    print("\n" + "="*65)
    print("STARTING DLINEAR TRAINING (ENDOGENOUS-ONLY BASELINE)")
    print("="*65)

    training_start_time = time.time()

    # Step 7: Training Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()
        train_losses = []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            # Step 4: Extract ONLY historical OT (last column)
            x_target = batch_x[:, :, -1:].float().to(device)  # [B, 96, 1]
            y_target = batch_y[:, -args.pred_len:, -1:].float().to(device)  # [B, 96, 1]

            optimizer.zero_grad()
            pred = model(x_target)
            loss = criterion(pred, y_target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                x_target = batch_x[:, :, -1:].float().to(device)
                y_target = batch_y[:, -args.pred_len:, -1:].float().to(device)
                pred = model(x_target)
                v_loss = criterion(pred, y_target)
                val_losses.append(v_loss.item())

        val_loss = float(np.mean(val_losses))
        epoch_time = time.time() - epoch_start_time

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "epoch_runtime_sec": epoch_time
        })

        print(f"Epoch: {epoch:02d} | Train MSE: {train_loss:.7f} | Val MSE: {val_loss:.7f} | Time: {epoch_time:.2f}s")

        # Checkpoint selection based on validation MSE
        if val_loss < best_val_loss:
            print(f"  --> Validation loss decreased ({best_val_loss:.7f} --> {val_loss:.7f}). Saving checkpoint...")
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            patience_counter += 1
            print(f"  --> EarlyStopping counter: {patience_counter} out of {PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}!")
                early_stopped = True
                break

    total_train_time = time.time() - training_start_time
    actual_epochs = len(history)

    # Step 8: Print epoch history table
    print("\n" + "="*65)
    print("DLINEAR TRAINING HISTORY")
    print("="*65)
    print(f"{'Epoch':<8}{'Train MSE':<18}{'Val MSE':<18}{'Runtime (s)':<12}")
    print("-" * 56)
    for row in history:
        print(f"{row['epoch']:<8}{row['train_mse']:<18.7f}{row['val_mse']:<18.7f}{row['epoch_runtime_sec']:<12.2f}")
    print("-" * 56)
    print(f"Planned maximum epochs:  {MAX_EPOCHS}")
    print(f"Actual epochs completed: {actual_epochs}")
    print(f"Early stopping triggered:{early_stopped}")
    print(f"Best validation epoch:   {best_epoch}")
    print(f"Best validation MSE:     {best_val_loss:.7f}")
    print(f"Total training runtime:  {total_train_time:.2f}s")

    # Step 9: Reload best checkpoint
    print("\n>>> Reloading best checkpoint for evaluation...")
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    print(f"Successfully reloaded best checkpoint from: {best_ckpt_path}")

    # Step 10: Final test evaluation
    print("\n>>> Evaluating on official test set...")
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            x_target = batch_x[:, :, -1:].float().to(device)
            y_target = batch_y[:, -args.pred_len:, -1:].float().to(device)

            pred = model(x_target)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(y_target.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # [10432, 96, 1]
    trues = np.concatenate(trues_list, axis=0)  # [10432, 96, 1]

    evaluated_windows = preds.shape[0]
    expected_windows = 10432
    assert evaluated_windows == expected_windows, f"Window mismatch: {evaluated_windows} != {expected_windows}"
    assert preds.shape == (expected_windows, 96, 1)
    assert trues.shape == (expected_windows, 96, 1)

    assert np.all(np.isfinite(preds)), "Predictions contain non-finite values!"
    assert np.all(np.isfinite(trues)), "Ground truth contains non-finite values!"
    nan_count = int(np.isnan(preds).sum())
    inf_count = int(np.isinf(preds).sum())
    assert nan_count == 0, "Predictions contain NaNs!"
    assert inf_count == 0, "Predictions contain Infs!"

    mae, mse, rmse, mape, mspe, rse, corr, nse, kge, r2 = metric(preds, trues)
    corr_scalar = float(np.mean(corr)) if isinstance(corr, np.ndarray) else float(corr)

    print("\n" + "="*50)
    print("DLINEAR BASELINE — WEATHER 9696 TEST RESULTS")
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

    # Step 11: Comparison across all three models
    pers_mse = 0.0012442638
    pers_mae = 0.0248438435

    xlin_mse = 0.0013196687
    xlin_mae = 0.0264768731

    dlin_mse = float(mse)
    dlin_mae = float(mae)

    # XLinear improvement relative to DLinear
    xlin_rel_mse_imp = ((dlin_mse - xlin_mse) / dlin_mse) * 100.0
    xlin_rel_mae_imp = ((dlin_mae - xlin_mae) / dlin_mae) * 100.0

    # DLinear vs Persistence difference
    dlin_vs_pers_mse_imp = ((pers_mse - dlin_mse) / pers_mse) * 100.0
    dlin_vs_pers_mae_imp = ((pers_mae - dlin_mae) / pers_mae) * 100.0

    print("\n" + "="*65)
    print("THREE-MODEL COMPARISON (WEATHER 96 -> 96)")
    print("="*65)
    print(f"{'Model':<16}{'MSE':<18}{'MAE':<18}")
    print("-" * 52)
    print(f"{'Persistence':<16}{pers_mse:<18.7f}{pers_mae:<18.7f}")
    print(f"{'DLinear':<16}{dlin_mse:<18.7f}{dlin_mae:<18.7f}")
    print(f"{'XLinear':<16}{xlin_mse:<18.7f}{xlin_mae:<18.7f}")
    print("-" * 52)

    print("\nXLinear Improvement Relative to DLinear:")
    print(f"  MSE Improvement (%): {xlin_rel_mse_imp:+.4f}%")
    print(f"  MAE Improvement (%): {xlin_rel_mae_imp:+.4f}%")

    print("\nDLinear Difference Relative to Persistence:")
    print(f"  MSE Improvement (%): {dlin_vs_pers_mse_imp:+.4f}%")
    print(f"  MAE Improvement (%): {dlin_vs_pers_mae_imp:+.4f}%")

    # Step 12: Efficiency
    xlin_params = 1348860
    param_reduction = ((xlin_params - total_params) / xlin_params) * 100.0
    ckpt_size = os.path.getsize(best_ckpt_path)

    print("\n" + "="*50)
    print("EFFICIENCY METRICS")
    print("="*50)
    print(f"DLinear Total Parameters:       {total_params:,}")
    print(f"DLinear Trainable Parameters:   {trainable_params:,}")
    print(f"XLinear Total Parameters:       {xlin_params:,}")
    print(f"Parameter Reduction vs XLinear: {param_reduction:.2f}%")
    print(f"Checkpoint File Size:           {ckpt_size:,} bytes")
    print(f"Total Training Runtime:         {total_train_time:.2f}s")
    print(f"Best Checkpoint Epoch:          {best_epoch}")

    # Step 13: Save artifacts
    pred_path = os.path.join('baseline_results', 'dlinear_weather_96_predictions.npy')
    true_path = os.path.join('baseline_results', 'dlinear_weather_96_truth.npy')
    metrics_json_path = os.path.join('baseline_results', 'dlinear_weather_96_metrics.json')
    history_json_path = os.path.join('baseline_results', 'dlinear_weather_96_history.json')

    np.save(pred_path, preds)
    np.save(true_path, trues)

    metrics_payload = {
        "model": "DLinear",
        "baseline_type": "endogenous-only",
        "dataset": "Weather",
        "target": "OT",
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "moving_avg": 25,
        "seed": SEED,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "actual_epochs": actual_epochs,
        "best_epoch": best_epoch,
        "parameters": total_params,
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
        "xlinear_improvement_relative_to_dlinear_mse_percent": float(xlin_rel_mse_imp),
        "xlinear_improvement_relative_to_dlinear_mae_percent": float(xlin_rel_mae_imp)
    }

    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)

    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    total_elapsed = time.time() - total_start_time
    print(f"\nSaved all artifacts to baseline_results/")
    print(f"Total script runtime: {total_elapsed:.2f}s")


if __name__ == '__main__':
    main()
