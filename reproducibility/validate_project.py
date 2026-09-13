import os
import sys
import hashlib
import json

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    overall_pass = True

    def fail(msg):
        nonlocal overall_pass
        print(f"FAIL: {msg}")
        overall_pass = False

    def pass_msg(msg):
        print(f"PASS: {msg}")

    # ==================== Environment ====================
    print("=== Environment ===")
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    if py_major == 3 and py_minor == 11:
        pass_msg(f"Python version is 3.11 ({sys.version.split()[0]})")
    else:
        fail(f"Python version expected 3.11, found {py_major}.{py_minor}")

    modules = ['torch', 'numpy', 'pandas', 'sklearn', 'scipy', 'streamlit']
    for mod in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', 'available')
            pass_msg(f"Module '{mod}' imported successfully ({ver})")
        except ImportError as e:
            fail(f"Module '{mod}' import failed: {e}")

    # ==================== Dataset ====================
    print("\n=== Dataset ===")
    dataset_path = os.path.join(repo_root, 'dataset', 'weather.csv')
    if not os.path.isfile(dataset_path):
        fail(f"dataset/weather.csv not found at {dataset_path}")
    else:
        pass_msg("dataset/weather.csv exists")
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            rows, cols = df.shape
            if rows == 52696:
                pass_msg(f"Dataset row count matches expected: {rows}")
            else:
                fail(f"Dataset row count expected 52696, got {rows}")

            if cols == 22:
                pass_msg(f"Dataset column count matches expected: {cols}")
            else:
                fail(f"Dataset column count expected 22, got {cols}")

            target_col = df.columns[-1]
            if target_col == 'OT':
                pass_msg(f"Final column is target variable '{target_col}'")
            else:
                fail(f"Expected final column 'OT', got '{target_col}'")

            non_date_cols = len(df.columns) - 1
            if non_date_cols == 21:
                pass_msg(f"Non-date feature variables: {non_date_cols}")
            else:
                fail(f"Expected 21 non-date variables, got {non_date_cols}")

            missing_count = int(df.isnull().sum().sum())
            if missing_count == 0:
                pass_msg("Zero missing / NaN values across all cells")
            else:
                fail(f"Found {missing_count} missing values in dataset")

            try:
                date_series = pd.to_datetime(df.iloc[:, 0])
                if len(date_series) == rows:
                    pass_msg(f"All {rows} date timestamps parsed successfully")
                else:
                    fail("Date parsing length mismatch")
            except Exception as pe:
                fail(f"Failed to parse dates: {pe}")
        except Exception as e:
            fail(f"Error validating dataset: {e}")

    # ==================== Predictions ====================
    print("\n=== Predictions ===")
    import numpy as np

    prediction_targets = [
        ('MS', 'results/weather_96_96_XLinear_custom_ftMS_sl96_ll48_pl96_dm1024_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy', (10432, 96, 1)),
        ('Full M', 'results/weather_96_96_XLinear_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy', (10432, 96, 21)),
        ('ES', 'results/weather_96_96_XLinear-ES_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy', (10432, 96, 21)),
        ('GT', 'results/weather_96_96_XLinear-GT_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy', (10432, 96, 21)),
        ('Persistence', 'baseline_results/weather_persistence_96_predictions.npy', (10432, 96, 1)),
        ('DLinear', 'baseline_results/dlinear_weather_96_predictions.npy', (10432, 96, 1)),
    ]

    for label, rel_path, expected_shape in prediction_targets:
        full_path = os.path.join(repo_root, rel_path.replace('/', os.sep))
        if not os.path.isfile(full_path):
            fail(f"{label} predictions file missing: {rel_path}")
            continue

        try:
            arr = np.load(full_path)
            shape_match = (arr.shape == expected_shape)
            is_numeric = np.issubdtype(arr.dtype, np.number)
            all_finite = bool(np.all(np.isfinite(arr)))
            nans = int(np.isnan(arr).sum())
            infs = int(np.isinf(arr).sum())

            if shape_match and is_numeric and all_finite and nans == 0 and infs == 0:
                pass_msg(f"{label:12s} shape={arr.shape}, dtype={arr.dtype}, finite=True, NaN=0, Inf=0")
            else:
                fail(f"{label:12s} validation failed: shape={arr.shape} (exp {expected_shape}), finite={all_finite}, NaN={nans}, Inf={infs}")
        except Exception as e:
            fail(f"Error loading {label} predictions from {rel_path}: {e}")

    # ==================== Checkpoints ====================
    print("\n=== Checkpoints ===")
    checkpoint_targets = [
        ('MS Checkpoint', 'checkpoints/weather_96_96_XLinear_custom_ftMS_sl96_ll48_pl96_dm1024_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'),
        ('Full M Checkpoint', 'checkpoints/weather_96_96_XLinear_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'),
        ('ES Checkpoint', 'checkpoints/weather_96_96_XLinear-ES_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'),
        ('GT Checkpoint', 'checkpoints/weather_96_96_XLinear-GT_custom_ftM_sl96_ll48_pl96_dm256_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth'),
        ('DLinear Checkpoint', 'baseline_results/dlinear_weather_96_checkpoint.pth')
    ]

    for label, rel_path in checkpoint_targets:
        full_path = os.path.join(repo_root, rel_path.replace('/', os.sep))
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            if size > 0:
                pass_msg(f"{label:18s} exists and non-empty ({size:,} bytes)")
            else:
                fail(f"{label:18s} is empty (0 bytes)")
        else:
            fail(f"{label:18s} missing: {rel_path}")

    # ==================== Analysis ====================
    print("\n=== Analysis ===")
    import pandas as pd

    csv_files = [
        ('weather_ablation_summary.csv', 'analysis/results/weather_ablation_summary.csv'),
        ('weather_ablation_tradeoff.csv', 'analysis/results/weather_ablation_tradeoff.csv')
    ]
    for label, rel_path in csv_files:
        full_path = os.path.join(repo_root, rel_path.replace('/', os.sep))
        if os.path.isfile(full_path):
            try:
                df = pd.read_csv(full_path)
                pass_msg(f"{label} parseable ({len(df)} rows, {len(df.columns)} cols)")
            except Exception as e:
                fail(f"{label} failed to parse: {e}")
        else:
            fail(f"{label} missing: {rel_path}")

    evidence_path = os.path.join(repo_root, 'analysis', 'results', 'weather_ablation_evidence.json')
    if os.path.isfile(evidence_path):
        try:
            with open(evidence_path, 'r', encoding='utf-8') as f:
                ev = json.load(f)
            pass_msg(f"weather_ablation_evidence.json parseable ({len(ev)} top-level sections)")
        except Exception as e:
            fail(f"weather_ablation_evidence.json parse error: {e}")
    else:
        fail("analysis/results/weather_ablation_evidence.json missing")

    findings_path = os.path.join(repo_root, 'analysis', 'results', 'weather_ablation_findings.md')
    if os.path.isfile(findings_path) and os.path.getsize(findings_path) > 0:
        pass_msg(f"weather_ablation_findings.md exists ({os.path.getsize(findings_path):,} bytes)")
    else:
        fail("analysis/results/weather_ablation_findings.md missing or empty")

    png_plots = [
        'weather_ablation_local_metrics.png',
        'weather_ablation_mse_vs_paper.png',
        'weather_ablation_mae_vs_paper.png',
        'weather_ablation_parameters.png',
        'weather_ablation_runtime.png'
    ]
    for plot_name in png_plots:
        full_path = os.path.join(repo_root, 'analysis', 'results', plot_name)
        if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
            pass_msg(f"{plot_name:36s} exists ({os.path.getsize(full_path):,} bytes)")
        else:
            fail(f"Plot missing or empty: {plot_name}")

    # ==================== Dashboard ====================
    print("\n=== Dashboard ===")
    dash_app = os.path.join(repo_root, 'dashboard', 'app.py')
    dash_readme = os.path.join(repo_root, 'dashboard', 'README.md')

    if os.path.isfile(dash_app) and os.path.getsize(dash_app) > 0:
        pass_msg(f"dashboard/app.py exists ({os.path.getsize(dash_app):,} bytes)")
    else:
        fail("dashboard/app.py missing or empty")

    if os.path.isfile(dash_readme) and os.path.getsize(dash_readme) > 0:
        pass_msg(f"dashboard/README.md exists ({os.path.getsize(dash_readme):,} bytes)")
    else:
        fail("dashboard/README.md missing or empty")

    # ==================== Checksums ====================
    print("\n=== Checksums ===")
    checksum_file = os.path.join(repo_root, 'reproducibility', 'checksums.sha256')
    if not os.path.isfile(checksum_file):
        fail(f"reproducibility/checksums.sha256 not found: {checksum_file}")
    else:
        with open(checksum_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                fail(f"Malformed checksum line: {line}")
                continue
            expected_hash, rel_path = parts
            full_path = os.path.join(repo_root, rel_path.replace('/', os.sep))
            if not os.path.isfile(full_path):
                fail(f"FAIL {rel_path} (file not found)")
                continue

            h = hashlib.sha256()
            with open(full_path, 'rb') as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_hash = h.hexdigest()

            if actual_hash == expected_hash:
                pass_msg(f"{rel_path}")
            else:
                fail(f"{rel_path} (expected {expected_hash}, got {actual_hash})")

    # ==================== Verdict ====================
    print()
    if overall_pass:
        print("PROJECT REPRODUCIBILITY VALIDATION: PASS")
        sys.exit(0)
    else:
        print("PROJECT REPRODUCIBILITY VALIDATION: FAIL")
        sys.exit(1)

if __name__ == '__main__':
    main()
