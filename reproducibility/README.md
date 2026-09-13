# XLinear Project Reproducibility

## Purpose

This reproducibility package verifies and validates the integrity of previously generated research artifacts, experimental outputs, local baseline results, ablation studies, and demonstration files without triggering model retraining. It ensures that all recorded benchmarks, prediction arrays, and weights remain consistent, finite, and cryptographically verified against their SHA-256 signatures.

## Validate

From the repository root with the project virtual environment active:

```powershell
python reproducibility/validate_project.py
```

## What It Checks

1. **Environment:** Python 3.11 runtime, imports and version availability of PyTorch, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, and Streamlit.
2. **Weather Dataset Structure:** Existence of `dataset/weather.csv`, row count (52,696), column count (22), target column (`OT`), 21 exogenous features, absence of missing values, and valid timestamp parsing.
3. **Prediction Arrays:** Verifies that all 6 prediction `.npy` files exist, match expected tensor shapes (`(10432, 96, 1)` or `(10432, 96, 21)`), are numeric, completely finite, and contain zero `NaN` or `Inf` values.
4. **Checkpoints:** Confirms existence and non-zero byte size of all 4 XLinear checkpoints (`MS`, `Full M`, `ES`, `GT`) and the DLinear baseline checkpoint.
5. **Ablation Analysis Artifacts:** Validates structure and readability of `weather_ablation_summary.csv`, `weather_ablation_tradeoff.csv`, `weather_ablation_evidence.json`, `weather_ablation_findings.md`, and all five 200 DPI PNG figures.
6. **Dashboard Files:** Confirms presence and non-zero size of `dashboard/app.py` and `dashboard/README.md`.
7. **SHA-256 Integrity:** Recalculates binary SHA-256 hashes across immutable research artifacts, dataset, checkpoints, prediction arrays, analysis records, and demo code against `reproducibility/checksums.sha256`.

## What It Does Not Do

* Does not retrain or fine-tune any models.
* Does not reproduce GPU training runtime or GPU execution timings.
* Does not download or modify benchmark datasets.
* Does not mutate, overwrite, or delete existing experimental artifacts.
* Does not guarantee bit-for-bit floating point equivalence across disparate operating systems or hardware architectures.

## Environment Note

All reported local experiments and reproductions were executed in a controlled local Windows environment using:

* **Operating System:** Windows 11 (AMD64)
* **Python Runtime:** Python 3.11.4
* **Deep Learning Framework:** PyTorch 2.6.0 CPU build (`torch.cuda.is_available() == False`)
* **Dataloader Concurrency:** `num_workers=0` (environment-specific adjustment to ensure stable, crash-free multiprocessing under Windows)
