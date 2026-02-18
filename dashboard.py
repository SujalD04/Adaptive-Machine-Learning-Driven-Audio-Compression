import streamlit as st
from pathlib import Path
from dashboard_engine import run_audio_simulation, load_model


st.set_page_config(page_title="Adaptive Opus Dashboard", layout="wide")


st.title("Adaptive Opus — Controller Comparison Dashboard")

st.markdown("Upload an audio file (wav/flac). The app will run four controllers and show MOS and playback.")

uploaded = st.file_uploader("Upload audio (.wav or .flac)", type=["wav", "flac"])
packet_loss = st.slider("Simulated packet loss (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

model = load_model()
if model is None:
    st.warning("ML model not found or failed to load. 'ML-Adaptive' and 'Hybrid' will use fallback behavior.")

if uploaded is not None:
    # Save uploaded file to a temporary path inside project
    tmp_dir = Path.home() / "adaptive_opus" / "uploaded"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / uploaded.name
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.audio(tmp_path)

    if st.button("Run Simulation"):
        st.info("Running simulation across controllers — this may take several seconds.")

        tabs = st.tabs(["Static", "Heuristic", "ML-Adaptive", "Hybrid"])
        controllers = ["static", "heuristic", "ml_adaptive", "hybrid"]

        for tab, ctrl in zip(tabs, controllers):
            with tab:
                st.write(f"Running: {ctrl}")
                metrics = run_audio_simulation(tmp_path, packet_loss, ctrl, model)
                if not metrics:
                    st.error("Simulation failed for this controller.")
                    continue

                st.metric("MOS (PESQ)", f"{metrics['mos']:.3f}")
                st.write("**Selected Configuration**")
                st.json(metrics["config"])

                st.write("**Processing time (s)**: ", f"{metrics['processing_time_s']:.2f}")

                # Playback processed audio
                try:
                    with open(metrics['processed_file'], 'rb') as pf:
                        st.audio(pf.read(), format='audio/wav')
                except Exception as e:
                    st.write("Processed audio unavailable for playback.")

        st.success("All controllers completed.")

else:
    st.info("Please upload an audio file to begin.")
