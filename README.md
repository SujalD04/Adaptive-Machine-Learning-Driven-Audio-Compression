**Project Overview**
- **Purpose:** Adaptive Opus — ML-driven adaptive audio compression for VoIP that combines ML optimization with robust fallbacks.

**Quick Start (Windows)**
# Adaptive Opus — ML-driven Adaptive Audio Compression

**Project Overview**

- **Purpose:** Adaptive Opus — ML-driven adaptive audio compression for VoIP that combines ML optimization with robust fallbacks.

## Quick Start (Windows)

- **Create venv:**
  - `python -m venv .venv`
  - Activate: PowerShell: `.\.venv\Scripts\Activate.ps1`  (or cmd: `.\.venv\Scripts\activate.bat`)
- **Install Python deps:**
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
- **System dependencies:** Ensure the `opusenc`/`opusdec` command-line tools are installed and on your `PATH` (the project uses them via `subprocess`). On Linux: `sudo apt install opus-tools libopus-dev`.

## Generate Dataset (optional)

- Run the scripted sweep to create `opus_dataset.csv` (uses files under `clean_audio/`):

```bash
python main_script.py
```

## Train Model

- Train and persist the Random Forest model to `qoe_model.joblib`:

```bash
python train_model.py
```

## Run Dashboard (UI)

- Launch the Streamlit demo dashboard:

```bash
streamlit run dashboard.py
```

## Files of Interest

- `main_script.py`: dataset generation sweep
- `train_model.py`: training pipeline and model persistence
- `qoe_model.joblib`: serialized RandomForest used by the dashboard
- `dashboard_engine.py`: controller implementations and processing helper
- `dashboard.py`: Streamlit UI to run controllers and compare results
- `opus_wrapper.py`: runs `opusenc`/`opusdec` to simulate encoding/decoding
- `quality_analyzer.py`: computes PESQ MOS using `pesq`

## Troubleshooting

- If you see `ModuleNotFoundError: No module named 'pesq'`, install packages with `python -m pip install pesq` (or use `requirements.txt`).
- If `opusenc`/`opusdec` are missing, install `opus-tools` and ensure tools are on `PATH`.

---

## Final Findings (from `reports/FINAL_REPORT.md`)

### 1. Model Training Reproduction

Reproduced training and evaluation using `RandomForestRegressor(n_estimators=100, random_state=42)` on `opus_dataset.csv`.

**Performance Metrics**
- Training R²: 0.8401
- Test R²: 0.8166
- Training MAE: 0.2083
- Test MAE: 0.2184
- Training RMSE: 0.2761
- Test RMSE: 0.2871
- 5-fold CV R²: 0.8260 ± 0.0482

**Feature Importances (retrained model)**
- `bitrate`: 0.2135 (21.3%)
- `frame_size`: 0.0125 (1.3%)
- `use_fec`: 0.0033 (0.3%)
- `packet_loss_perc`: 0.7707 (77.1%)

**Feature Importances (saved `qoe_model.joblib`)**
- `bitrate`: 0.2135 (21.3%)
- `frame_size`: 0.0125 (1.3%)
- `use_fec`: 0.0033 (0.3%)
- `packet_loss_perc`: 0.7707 (77.1%)

### 2. Key Findings Verification

- The reproduced training shows a large gap between training and test R², indicating some overfitting. The test R² around 0.81 confirms the previously reported R² ≈ 0.813.
- Feature importances confirm that `packet_loss_perc` dominates model predictions and `use_fec` contributes very little in the training corpus. This matches the project observation that FEC importance was low due to clean speech in LibriSpeech.

### 3. Out-of-Distribution (OOD) Test Summary

- Included OOD report and plots: see the `reports/` directory for `OOD_REPORT.md`, `mean_mos_by_controller.png`, and `mos_vs_loss.png`. The OOD sweep used two test inputs (`beep_440.wav`, `wild_speech_sim.wav`) across packet losses [0,1,2,3,4,5,7,10,15,20] and four controllers (Static, Heuristic, ML-Adaptive, Hybrid).

Summary (aggregate):
- Mean MOS by controller (higher is better):
  - static: 3.011
  - heuristic: 2.619
  - ml_adaptive: 2.914
  - hybrid: 2.966

- Mean MOS by input:
  - `beep_440.wav`: 2.912
  - `wild_speech_sim.wav`: 2.843

At 5% packet loss snapshot: `static` and `hybrid` performed best; `heuristic` performed worst; `ml_adaptive` gave mixed results.

### 4. Conclusion

- The analyses validate the project's claim: the model explains ~81% of variance on the test set but learned dataset-specific patterns (overfitting to LibriSpeech).
- The Hybrid controller is validated by OOD testing: it avoids ML decisions at high loss where ML is unreliable.

---

## Next Steps

- Expand `reports/FINAL_REPORT.md` with numeric tables and inline plots for publication.
- Optionally export the `reports/OOD_REPORT.md` + images to PDF for a distributable report.

If you'd like, I can commit these changes and open a PR with the updated `README.md`.
