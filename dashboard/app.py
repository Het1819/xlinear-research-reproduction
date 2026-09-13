import os
import sys
import time
from types import SimpleNamespace
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import XLinear
from data_provider.data_factory import data_provider

# Page configuration
st.set_page_config(
    page_title="XLinear Weather Forecasting Demo",
    page_icon="⏱️",
    layout="wide"
)

# Constants
CHECKPOINT_SETTING = "weather_96_96_XLinear_custom_ftMS_sl96_ll48_pl96_dm1024_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0"
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "checkpoints", CHECKPOINT_SETTING, "checkpoint.pth")
DATASET_PATH = os.path.join(REPO_ROOT, "dataset", "weather.csv")

@st.cache_resource
def load_cached_model():
    if not os.path.exists(CHECKPOINT_PATH):
        st.error(f"Checkpoint file not found: {CHECKPOINT_PATH}")
        st.stop()

    configs = SimpleNamespace(
        seq_len=96,
        pred_len=96,
        enc_in=21,
        d_model=1024,
        t_ff=256,
        c_ff=21,
        usenorm=1,
        embed_dropout=0.1,
        head_dropout=0.0,
        t_dropout=0.0,
        c_dropout=0.0,
        features='MS'
    )

    try:
        model = XLinear.Model(configs)
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    assert next(model.parameters()).device.type == 'cpu', "Model is not on CPU!"
    assert not model.training, "Model is in training mode!"

    return model

@st.cache_resource
def load_cached_test_dataset():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at: {DATASET_PATH}")
        st.stop()

    args = SimpleNamespace(
        data='custom',
        root_path=os.path.join(REPO_ROOT, 'dataset/'),
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

    try:
        test_data, _ = data_provider(args, flag='test')
    except Exception as e:
        st.error(f"Failed to load test dataset: {e}")
        st.stop()

    if len(test_data) < 10432:
        st.error(f"Test dataset size unexpectedly small: {len(test_data)} < 10432")
        st.stop()

    return test_data

def main():
    st.title("XLinear Weather Forecasting Demo")
    st.subheader("AAAI 2026 XLinear Reproduction — Exogenous Time-Series Forecasting")

    st.info(
        "This demonstration uses a held-out Weather benchmark test window. "
        "XLinear receives 96 historical observations from 21 variables and "
        "forecasts the next 96 values of the target variable OT."
    )
    st.caption("Retrospective evaluation demo — not a live weather forecasting service.")

    # Sidebar
    st.sidebar.header("Controls & Metadata")
    sample_index = st.sidebar.slider(
        "Sample index",
        min_value=0,
        max_value=10431,
        value=0,
        step=1,
        help="Select a window from the held-out evaluation population (0-10431)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Static Model Architecture:**")
    st.sidebar.markdown("- **Model:** XLinear")
    st.sidebar.markdown("- **Task:** MS — multivariate input / univariate output")
    st.sidebar.markdown("- **History:** 96 steps")
    st.sidebar.markdown("- **Forecast horizon:** 96 steps")
    st.sidebar.markdown("- **Input variables:** 21")
    st.sidebar.markdown("- **Target:** OT")
    st.sidebar.markdown("- **Device:** CPU")

    # Load model and dataset
    model = load_cached_model()
    test_data = load_cached_test_dataset()

    if sample_index < 0 or sample_index >= 10432:
        st.error(f"Sample index {sample_index} is out of allowable range [0, 10431]")
        st.stop()

    # Retrieve sample
    seq_x, seq_y, seq_x_mark, seq_y_mark = test_data[sample_index]

    if seq_x.shape != (96, 21) or seq_y.shape != (144, 21):
        st.error(f"Unexpected tensor shape: seq_x {seq_x.shape}, seq_y {seq_y.shape}")
        st.stop()

    if not np.all(np.isfinite(seq_x)) or not np.all(np.isfinite(seq_y)):
        st.error("Non-finite values detected in input sample data.")
        st.stop()

    x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)  # [1, 96, 21]

    # Run inference with timing
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_out = model(x)
    inference_time_ms = (time.perf_counter() - t0) * 1000.0

    if pred_out.shape != (1, 96, 1):
        st.error(f"Unexpected model output shape: {pred_out.shape} != (1, 96, 1)")
        st.stop()

    pred_scaled = pred_out[0, :, 0].numpy()
    true_scaled = seq_y[-96:, -1]

    if not np.all(np.isfinite(pred_scaled)):
        st.error("Model produced non-finite predictions.")
        st.stop()

    # Compute scaled metrics
    scaled_mse = float(np.mean((pred_scaled - true_scaled) ** 2))
    scaled_mae = float(np.mean(np.abs(pred_scaled - true_scaled)))

    # Convert to original dataset scale
    target_mean = float(test_data.scaler.mean_[-1])
    target_scale = float(test_data.scaler.scale_[-1])

    history_ot_original = seq_x[:, -1] * target_scale + target_mean
    prediction_original = pred_scaled * target_scale + target_mean
    truth_original = true_scaled * target_scale + target_mean

    if not (np.all(np.isfinite(history_ot_original)) and
            np.all(np.isfinite(prediction_original)) and
            np.all(np.isfinite(truth_original))):
        st.error("Non-finite values generated during scale inversion.")
        st.stop()

    original_mae = float(np.mean(np.abs(prediction_original - truth_original)))

    # Metrics Display
    st.markdown("### Test Window Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Scaled MSE", f"{scaled_mse:.6f}")
    col2.metric("Scaled MAE", f"{scaled_mae:.6f}")
    col3.metric("Original-Scale MAE", f"{original_mae:.4f}")
    st.caption("These metrics apply only to the selected held-out test window.")

    # Visualization
    st.markdown("### Forecast Visualization")
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)

    x_history = np.arange(-95, 1)    # -95 to 0 (96 points)
    x_forecast = np.arange(1, 97)   # 1 to 96 (96 points)

    ax.plot(x_history, history_ot_original, label='Historical OT', color='#1f77b4', linewidth=1.8)
    ax.plot(x_forecast, truth_original, label='Actual Future OT', color='#2ca02c', linewidth=1.8, linestyle='--')
    ax.plot(x_forecast, prediction_original, label='XLinear Forecast', color='#d62728', linewidth=2.0)

    ax.axvline(x=0, color='#7f7f7f', linestyle=':', linewidth=1.5, label='Forecast Origin (t=0)')

    ax.set_title("Historical Target and 96-Step XLinear Forecast", fontsize=13, pad=10)
    ax.set_xlabel("Relative timestep", fontsize=11)
    ax.set_ylabel("OT — original dataset scale", fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

    # Expandable Technical Information
    with st.expander("Technical details"):
        st.markdown(f"- **Checkpoint:** `{os.path.basename(CHECKPOINT_PATH)}`")
        total_params = sum(p.numel() for p in model.parameters())
        st.markdown(f"- **Setting directory:** `{CHECKPOINT_SETTING}`")
        st.markdown(f"- **Model parameters:** {total_params:,}")
        st.markdown("- **Input tensor:** `(1, 96, 21)`")
        st.markdown("- **Output tensor:** `(1, 96, 1)`")
        st.markdown("- **Target variable:** `OT`")
        st.markdown("- **Training random seed:** 2025")
        st.markdown("- **Local reproduction metrics (overall test set):** MSE = `0.0013196687`, MAE = `0.0264768731`")
        st.markdown("- **Paper reported Weather 96 (Table 2):** MSE = `0.001`, MAE = `0.026`")
        st.markdown(f"- **Observed local CPU inference time:** `{inference_time_ms:.2f} ms`")
        st.caption("Local values are reproduction results from this environment and are not guaranteed across hardware/software configurations.")

if __name__ == '__main__':
    main()
