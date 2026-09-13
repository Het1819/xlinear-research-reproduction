import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 2: Verify existing experiment artifacts
    # ---------------------------------------------------------
    settings = {
        'Full XLinear': 'weather_96_96_XLinear_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0',
        'XLinear-ES': 'weather_96_96_XLinear-ES_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0',
        'XLinear-GT': 'weather_96_96_XLinear-GT_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0'
    }

    base_repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    print(">>> Verifying existing experiment artifacts...")
    for model_name, setting in settings.items():
        ckpt_path = os.path.join(base_repo_dir, 'checkpoints', setting, 'checkpoint.pth')
        pred_path = os.path.join(base_repo_dir, 'results', setting, 'pred.npy')

        assert os.path.exists(ckpt_path), f"Checkpoint missing for {model_name}: {ckpt_path}"
        assert os.path.exists(pred_path), f"Predictions missing for {model_name}: {pred_path}"

        pred_arr = np.load(pred_path)
        assert pred_arr.shape == (10432, 96, 21), f"Unexpected shape for {model_name}: {pred_arr.shape}"
        assert pred_arr.dtype == np.float32, f"Unexpected dtype for {model_name}: {pred_arr.dtype}"
        assert np.all(np.isfinite(pred_arr)), f"Non-finite values found in {model_name}"
        assert np.isnan(pred_arr).sum() == 0, f"NaNs found in {model_name}"
        assert np.isinf(pred_arr).sum() == 0, f"Infs found in {model_name}"
        print(f"  [OK] {model_name}: {pred_arr.shape}, finite, no NaN/Inf.")

    # ---------------------------------------------------------
    # STEP 3: Define Exact Records and Build Summary Table
    # ---------------------------------------------------------
    records = [
        {
            "model": "Full XLinear",
            "mse": 0.1509373039,
            "mae": 0.1989531070,
            "parameters": 345980,
            "training_runtime_seconds": 547.22,
            "epochs": 20,
            "best_epoch": 17,
            "paper_mse": 0.149,
            "paper_mae": 0.198
        },
        {
            "model": "XLinear-ES",
            "mse": 0.1768046767,
            "mae": 0.2164115608,
            "parameters": 181088,
            "training_runtime_seconds": 282.53,
            "epochs": 20,
            "best_epoch": 10,
            "paper_mse": 0.175,
            "paper_mae": 0.216
        },
        {
            "model": "XLinear-GT",
            "mse": 0.1537747383,
            "mae": 0.2020685524,
            "parameters": 321404,
            "training_runtime_seconds": 563.81,
            "epochs": 20,
            "best_epoch": 10,
            "paper_mse": 0.153,
            "paper_mae": 0.200
        }
    ]

    full_params = records[0]["parameters"]
    summary_rows = []

    for r in records:
        mse_abs_diff = r["mse"] - r["paper_mse"]
        mae_abs_diff = r["mae"] - r["paper_mae"]
        mse_pct_diff = (mse_abs_diff / r["paper_mse"]) * 100.0
        mae_pct_diff = (mae_abs_diff / r["paper_mae"]) * 100.0
        param_red_pct = ((full_params - r["parameters"]) / full_params) * 100.0
        avg_sec_epoch = r["training_runtime_seconds"] / r["epochs"]

        summary_rows.append({
            "model": r["model"],
            "local_mse": r["mse"],
            "local_mae": r["mae"],
            "paper_mse": r["paper_mse"],
            "paper_mae": r["paper_mae"],
            "mse_absolute_difference": mse_abs_diff,
            "mae_absolute_difference": mae_abs_diff,
            "mse_percent_difference_from_paper": mse_pct_diff,
            "mae_percent_difference_from_paper": mae_pct_diff,
            "parameters": r["parameters"],
            "parameter_reduction_vs_full_percent": param_red_pct,
            "training_runtime_seconds": r["training_runtime_seconds"],
            "average_seconds_per_epoch": avg_sec_epoch,
            "best_epoch": r["best_epoch"]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, 'weather_ablation_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f">>> Saved summary CSV to: {summary_csv_path}")

    # ---------------------------------------------------------
    # STEP 4: Calculate Ablation Effects
    # ---------------------------------------------------------
    full_mse = records[0]["mse"]
    full_mae = records[0]["mae"]
    es_mse = records[1]["mse"]
    es_mae = records[1]["mae"]
    gt_mse = records[2]["mse"]
    gt_mae = records[2]["mae"]

    es_mse_deg = ((es_mse - full_mse) / full_mse) * 100.0
    es_mae_deg = ((es_mae - full_mae) / full_mae) * 100.0

    gt_mse_deg = ((gt_mse - full_mse) / full_mse) * 100.0
    gt_mae_deg = ((gt_mae - full_mae) / full_mae) * 100.0

    gt_imp_es_mse = ((es_mse - gt_mse) / es_mse) * 100.0
    gt_imp_es_mae = ((es_mae - gt_mae) / es_mae) * 100.0

    print("\n" + "="*50)
    print("CALCULATED ABLATION EFFECTS")
    print("="*50)
    print(f"ES MSE degradation vs Full (%):     +{es_mse_deg:.4f}%")
    print(f"ES MAE degradation vs Full (%):     +{es_mae_deg:.4f}%")
    print(f"GT MSE degradation vs Full (%):     +{gt_mse_deg:.4f}%")
    print(f"GT MAE degradation vs Full (%):     +{gt_mae_deg:.4f}%")
    print(f"GT improvement over ES MSE (%):    +{gt_imp_es_mse:.4f}%")
    print(f"GT improvement over ES MAE (%):    +{gt_imp_es_mae:.4f}%")

    # ---------------------------------------------------------
    # STEP 5: Plot 1 - Local Ablation Accuracy (Grouped Bar Chart)
    # ---------------------------------------------------------
    plot1_models = ["Full XLinear", "XLinear-GT", "XLinear-ES"]
    plot1_mse = [full_mse, gt_mse, es_mse]
    plot1_mae = [full_mae, gt_mae, es_mae]

    x = np.arange(len(plot1_models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    rects1 = ax.bar(x - width/2, plot1_mse, width, label='MSE', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, plot1_mae, width, label='MAE', color='#d95f02')

    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Weather 9696 Multivariate Ablation — Local Results', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot1_models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(es_mse, es_mae) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'weather_ablation_local_metrics.png')
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f">>> Saved Plot 1: {plot1_path}")

    # ---------------------------------------------------------
    # STEP 6: Plot 2 - Local vs Paper (Separate MSE and MAE charts)
    # ---------------------------------------------------------
    # MSE chart
    paper_mse_vals = [0.149, 0.153, 0.175]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    r1 = ax.bar(x - width/2, plot1_mse, width, label='Local Exact', color='#1f77b4')
    r2 = ax.bar(x + width/2, paper_mse_vals, width, label='AAAI Paper', color='#aec7e8')
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_title('Weather 9696 Ablation — Local vs Paper MSE', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot1_models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(plot1_mse + paper_mse_vals) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    autolabel(r1)
    autolabel(r2)
    plt.tight_layout()
    plot2_mse_path = os.path.join(output_dir, 'weather_ablation_mse_vs_paper.png')
    plt.savefig(plot2_mse_path, dpi=200)
    plt.close()
    print(f">>> Saved Plot 2 (MSE): {plot2_mse_path}")

    # MAE chart
    paper_mae_vals = [0.198, 0.200, 0.216]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    r3 = ax.bar(x - width/2, plot1_mae, width, label='Local Exact', color='#d62728')
    r4 = ax.bar(x + width/2, paper_mae_vals, width, label='AAAI Paper', color='#ff9896')
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Weather 9696 Ablation — Local vs Paper MAE', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(plot1_models, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(plot1_mae + paper_mae_vals) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    autolabel(r3)
    autolabel(r4)
    plt.tight_layout()
    plot2_mae_path = os.path.join(output_dir, 'weather_ablation_mae_vs_paper.png')
    plt.savefig(plot2_mae_path, dpi=200)
    plt.close()
    print(f">>> Saved Plot 2 (MAE): {plot2_mae_path}")

    # ---------------------------------------------------------
    # STEP 7: Plot 3 - Parameter Efficiency
    # ---------------------------------------------------------
    plot3_models = ["Full XLinear", "XLinear-GT", "XLinear-ES"]
    plot3_params = [345980, 321404, 181088]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    bars = ax.bar(plot3_models, plot3_params, width=0.5, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    ax.set_ylabel('Parameters', fontsize=12)
    ax.set_title('Weather Ablation — Trainable Parameters', fontsize=14, pad=15)
    ax.set_ylim(0, max(plot3_params) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='semibold')

    plt.tight_layout()
    plot3_path = os.path.join(output_dir, 'weather_ablation_parameters.png')
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f">>> Saved Plot 3 (Parameters): {plot3_path}")

    # ---------------------------------------------------------
    # STEP 8: Plot 4 - Runtime Observation
    # ---------------------------------------------------------
    plot4_runtime = [547.22 / 20.0, 563.81 / 20.0, 282.53 / 20.0]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    bars4 = ax.bar(plot3_models, plot4_runtime, width=0.5, color=['#4c72b0', '#55a868', '#c44e52'])
    ax.set_ylabel('Seconds / epoch', fontsize=12)
    ax.set_title('Weather Ablation — Observed CPU Training Time per Epoch', fontsize=14, pad=15)
    ax.set_ylim(0, max(plot4_runtime) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars4:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='semibold')

    plt.tight_layout()
    plot4_path = os.path.join(output_dir, 'weather_ablation_runtime.png')
    plt.savefig(plot4_path, dpi=200)
    plt.close()
    print(f">>> Saved Plot 4 (Runtime): {plot4_path}")

    # ---------------------------------------------------------
    # STEP 9: Normalized Trade-off Table
    # ---------------------------------------------------------
    tradeoff_rows = []
    full_runtime_epoch = 547.22 / 20.0

    for r in records:
        sec_epoch = r["training_runtime_seconds"] / r["epochs"]
        tradeoff_rows.append({
            "model": r["model"],
            "mse": r["mse"],
            "mae": r["mae"],
            "parameters": r["parameters"],
            "mse_relative_to_full": r["mse"] / full_mse,
            "mae_relative_to_full": r["mae"] / full_mae,
            "parameters_relative_to_full": r["parameters"] / full_params,
            "seconds_per_epoch": sec_epoch,
            "seconds_per_epoch_relative_to_full": sec_epoch / full_runtime_epoch
        })

    tradeoff_df = pd.DataFrame(tradeoff_rows)
    tradeoff_csv_path = os.path.join(output_dir, 'weather_ablation_tradeoff.csv')
    tradeoff_df.to_csv(tradeoff_csv_path, index=False)
    print(f">>> Saved Trade-off CSV to: {tradeoff_csv_path}")

    # ---------------------------------------------------------
    # STEP 10, 11, 12: Findings Markdown
    # ---------------------------------------------------------
    findings_md_path = os.path.join(output_dir, 'weather_ablation_findings.md')
    findings_content = f"""# Weather 9696 Ablation Findings

## Experiment
* **Dataset:** Weather benchmark (`weather.csv`, 52,696 rows)
* **Task:** Multivariate time-series forecasting (`features='M'`) across all 21 variables
* **Sequence Length (`seq_len`):** 96
* **Prediction Horizon (`pred_len`):** 96
* **Input / Output Channels:** 21
* **Random Seed:** 2025
* **Execution Environment:** Windows 11 (AMD64), Python 3.11.4, PyTorch 2.6.0+cpu
* **Configuration Adjustment:** `--num_workers 0` applied strictly for Windows multiprocessing compatibility

## Local Results

| Model | MSE | MAE | Parameters | Seconds/Epoch |
| :--- | :---: | :---: | :---: | :---: |
| **Full XLinear** | {full_mse:.10f} | {full_mae:.10f} | 345,980 | {full_runtime_epoch:.2f}s |
| **XLinear-GT** | {gt_mse:.10f} | {gt_mae:.10f} | 321,404 | {(563.81/20.0):.2f}s |
| **XLinear-ES** | {es_mse:.10f} | {es_mae:.10f} | 181,088 | {(282.53/20.0):.2f}s |

## Paper Comparison

| Model | Local MSE | Paper MSE | Local MAE | Paper MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Full XLinear** | {full_mse:.4f} (~0.151) | 0.149 | {full_mae:.4f} (~0.199) | 0.198 |
| **XLinear-GT** | {gt_mse:.4f} (~0.154) | 0.153 | {gt_mae:.4f} (~0.202) | 0.200 |
| **XLinear-ES** | {es_mse:.4f} (~0.177) | 0.175 | {es_mae:.4f} (~0.216) | 0.216 |

*Note: The local CPU experimental values are close to the published numbers but are not strictly numerically identical.*

## Ablation Interpretation
1. Full XLinear produced the lowest local MSE ({full_mse:.4f}) and MAE ({full_mae:.4f}).
2. Removing the global/cross-variable contribution from the final representation (ES) produced the largest degradation: approximately +{es_mse_deg:.2f}% MSE and +{es_mae_deg:.2f}% MAE relative to Full XLinear.
3. GT remained much closer to Full XLinear: approximately +{gt_mse_deg:.2f}% MSE and +{gt_mae_deg:.2f}% MAE degradation.
4. GT substantially outperformed ES in this specific Weather 9696 experiment (+{gt_imp_es_mse:.2f}% MSE, +{gt_imp_es_mae:.2f}% MAE improvement).
5. This provides evidence that the global/cross-variable representation contains highly useful forecasting information for this dataset and horizon.
6. Full XLinear still outperformed GT, supporting the interpretation that the temporal and global/cross-variable representations provide complementary information.

*(Note: These empirical findings reflect this specific Weather 9696 experiment and should not be generalized to claim that cross-variable information is universally superior across all datasets or lookback configurations).*

## Reproduction Assessment
* **Paper vs. Local Comparison:**
  * **Full XLinear:** Paper MSE `0.149` vs. Local `~0.151`; Paper MAE `0.198` vs. Local `~0.199`.
  * **XLinear-ES:** Paper MSE `0.175` vs. Local `~0.177`; Paper MAE `0.216` vs. Local `~0.216`.
  * **XLinear-GT:** Paper MSE `0.153` vs. Local `~0.154`; Paper MAE `0.200` vs. Local `~0.202`.
* **Qualitative Order Consistency:**
  The local experiments reproduce the same qualitative ordering:
  $$\\text{{Full XLinear}} < \\text{{GT}} < \\text{{ES}}$$
  for both MSE and MAE (where "$<$" indicates lower prediction error).
* **Numerical Exactness:**
  Exact numerical equality was not achieved, but the results remain close to the reported AAAI values.

## Limitations
* Only the Weather dataset has been analyzed in this ablation series so far.
* Only horizon 96 in the ablation configuration has been locally tested.
* Experiments were executed on CPU rather than the GPU infrastructure used by the original authors.
* `num_workers` was adjusted to 0 for Windows runtime compatibility.
* Only a single fixed seed (2025) has been evaluated; multi-seed variance estimates and confidence intervals have not yet been established.
* Runtime measurements are machine-specific observations on local hardware and should not be interpreted as hardware-independent model benchmarks or compared directly to published GPU timings.
* MAPE is unstable and less interpretable for standardized data with values near zero, and is therefore not central to this benchmark comparison.
"""
    with open(findings_md_path, 'w', encoding='utf-8') as f:
        f.write(findings_content)
    print(f">>> Saved Findings Markdown to: {findings_md_path}")

    # ---------------------------------------------------------
    # STEP 13: Evidence JSON
    # ---------------------------------------------------------
    evidence = {
        "experiment": {
            "dataset": "Weather",
            "task": "multivariate forecasting",
            "seq_len": 96,
            "pred_len": 96,
            "channels": 21,
            "seed": 2025,
            "device": "CPU",
            "windows_evaluated": 10432
        },
        "local_results": {
            "XLinear": {
                "mse": full_mse,
                "mae": full_mae,
                "parameters": 345980,
                "training_runtime_seconds": 547.22,
                "epochs": 20,
                "best_epoch": 17
            },
            "XLinear-ES": {
                "mse": es_mse,
                "mae": es_mae,
                "parameters": 181088,
                "training_runtime_seconds": 282.53,
                "epochs": 20,
                "best_epoch": 10
            },
            "XLinear-GT": {
                "mse": gt_mse,
                "mae": gt_mae,
                "parameters": 321404,
                "training_runtime_seconds": 563.81,
                "epochs": 20,
                "best_epoch": 10
            }
        },
        "paper_results": {
            "XLinear": {
                "mse": 0.149,
                "mae": 0.198
            },
            "XLinear-ES": {
                "mse": 0.175,
                "mae": 0.216
            },
            "XLinear-GT": {
                "mse": 0.153,
                "mae": 0.200
            }
        },
        "ablation_effects": {
            "es_mse_degradation_vs_full_percent": es_mse_deg,
            "es_mae_degradation_vs_full_percent": es_mae_deg,
            "gt_mse_degradation_vs_full_percent": gt_mse_deg,
            "gt_mae_degradation_vs_full_percent": gt_mae_deg,
            "gt_improvement_over_es_mse_percent": gt_imp_es_mse,
            "gt_improvement_over_es_mae_percent": gt_imp_es_mae
        },
        "environment_deviations": [
            "num_workers=0 for Windows compatibility",
            "CPU execution rather than authors' GPU hardware"
        ]
    }

    evidence_json_path = os.path.join(output_dir, 'weather_ablation_evidence.json')
    with open(evidence_json_path, 'w', encoding='utf-8') as f:
        json.dump(evidence, f, indent=2)
    print(f">>> Saved Evidence JSON to: {evidence_json_path}")

    # ---------------------------------------------------------
    # STEP 14: Validation of all generated artifacts
    # ---------------------------------------------------------
    print("\n>>> Validating generated artifacts...")
    png_files = [
        plot1_path,
        plot2_mse_path,
        plot2_mae_path,
        plot3_path,
        plot4_path
    ]
    for p in png_files:
        assert os.path.exists(p), f"Missing plot: {p}"
        size = os.path.getsize(p)
        assert size > 0, f"Empty plot: {p}"
        print(f"  [OK] Plot verified: {os.path.basename(p)} ({size:,} bytes)")

    # Validate CSV reloads
    df_sum_check = pd.read_csv(summary_csv_path)
    assert len(df_sum_check) == 3, "Summary CSV rows != 3"
    print(f"  [OK] Summary CSV reloaded successfully ({len(df_sum_check)} rows).")

    df_trade_check = pd.read_csv(tradeoff_csv_path)
    assert len(df_trade_check) == 3, "Tradeoff CSV rows != 3"
    print(f"  [OK] Trade-off CSV reloaded successfully ({len(df_trade_check)} rows).")

    # Validate JSON reload
    with open(evidence_json_path, encoding='utf-8') as f:
        ev_check = json.load(f)
    assert "ablation_effects" in ev_check, "JSON missing ablation_effects"
    print(f"  [OK] Evidence JSON reloaded and validated.")

    print("\nAll artifacts generated and validated successfully!")

if __name__ == '__main__':
    main()
