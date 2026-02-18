**Project Overview**
- **Purpose:** Adaptive Opus — ML-driven adaptive audio compression for VoIP that combines ML optimization with robust fallbacks.

**Quick Start (Windows)**
- **Create venv:**
  - `python -m venv .venv`
  - Activate: PowerShell: `.\.venv\Scripts\Activate.ps1`  (or cmd: `.\.venv\Scripts\activate.bat`)
- **Install Python deps:**
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
- **System dependencies:** Ensure the `opusenc`/`opusdec` command-line tools are installed and on your `PATH` (the project uses them via `subprocess`). On Linux: `sudo apt install opus-tools libopus-dev`.

**Generate Dataset (optional)**
- Run the scripted sweep to create `opus_dataset.csv` (uses files under `clean_audio/`):
```bash
python main_script.py
```

**Train Model**
- Train and persist the Random Forest model to `qoe_model.joblib`:
```bash
python train_model.py
```

**Run Dashboard (UI)**
- Launch the Streamlit demo dashboard:
```bash
streamlit run dashboard.py
```

**Files of Interest**
- `main_script.py`: dataset generation sweep
- `train_model.py`: training pipeline and model persistence
- `qoe_model.joblib`: serialized RandomForest used by the dashboard
- `dashboard_engine.py`: controller implementations and processing helper
- `dashboard.py`: Streamlit UI to run controllers and compare results
- `opus_wrapper.py`: runs `opusenc`/`opusdec` to simulate encoding/decoding
- `quality_analyzer.py`: computes PESQ MOS using `pesq`

**Troubleshooting**
- If you see `ModuleNotFoundError: No module named 'pesq'`, install packages with `python -m pip install pesq` (or use `requirements.txt`).
- If `opusenc`/`opusdec` are missing, install `opus-tools` and ensure tools are on `PATH`.

**Next Steps I Can Do**
- Install Python packages into the workspace virtualenv and re-run a smoke test.
- Run the Streamlit dashboard here to validate end-to-end (requires system `opusenc`/`opusdec`).
