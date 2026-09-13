# Weather 9696 Ablation Findings

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
| **Full XLinear** | 0.1509373039 | 0.1989531070 | 345,980 | 27.36s |
| **XLinear-GT** | 0.1537747383 | 0.2020685524 | 321,404 | 28.19s |
| **XLinear-ES** | 0.1768046767 | 0.2164115608 | 181,088 | 14.13s |

## Paper Comparison

| Model | Local MSE | Paper MSE | Local MAE | Paper MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Full XLinear** | 0.1509 (~0.151) | 0.149 | 0.1990 (~0.199) | 0.198 |
| **XLinear-GT** | 0.1538 (~0.154) | 0.153 | 0.2021 (~0.202) | 0.200 |
| **XLinear-ES** | 0.1768 (~0.177) | 0.175 | 0.2164 (~0.216) | 0.216 |

*Note: The local CPU experimental values are close to the published numbers but are not strictly numerically identical.*

## Ablation Interpretation
1. Full XLinear produced the lowest local MSE (0.1509) and MAE (0.1990).
2. Removing the global/cross-variable contribution from the final representation (ES) produced the largest degradation: approximately +17.14% MSE and +8.78% MAE relative to Full XLinear.
3. GT remained much closer to Full XLinear: approximately +1.88% MSE and +1.57% MAE degradation.
4. GT substantially outperformed ES in this specific Weather 9696 experiment (+13.03% MSE, +6.63% MAE improvement).
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
  $$\text{Full XLinear} < \text{GT} < \text{ES}$$
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
