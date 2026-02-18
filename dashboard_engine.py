import time
from pathlib import Path
import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample

from opus_wrapper import process_audio_file, PROCESSED_AUDIO_DIR, BASE_PROJECT_DIR
from quality_analyzer import get_audio_quality


# Paths
BASE_DIR = Path.home() / "adaptive_opus"
MODEL_PATH = BASE_DIR / "qoe_model.joblib"


def resample_to_16khz(audio, sr):
    if sr == 16000:
        return audio, sr
    num_samples = int(len(audio) * 16000 / sr)
    audio_res = resample(audio, num_samples)
    return audio_res, 16000


def generate_all_configs():
    bitrates = [8, 12, 16, 24, 32, 48]
    frame_sizes = [20, 40, 60]
    complexities = [5, 9]
    fec_options = [False, True]

    configs = []
    for b in bitrates:
        for f in frame_sizes:
            for c in complexities:
                for fec in fec_options:
                    configs.append({
                        "bitrate": b,
                        "frame_size": f,
                        "complexity": c,
                        "use_fec": fec
                    })
    return configs


def static_controller(packet_loss):
    return {"bitrate": 32, "frame_size": 20, "complexity": 5, "use_fec": False}


def heuristic_controller(packet_loss):
    if packet_loss < 2:
        bitrate = 48
    elif packet_loss < 5:
        bitrate = 24
    elif packet_loss < 10:
        bitrate = 16
    else:
        bitrate = 12
    return {"bitrate": bitrate, "frame_size": 20, "complexity": 5, "use_fec": False}


def ml_controller(packet_loss, model):
    candidates = generate_all_configs()
    best = None
    best_score = -1e9

    # Prepare feature vector according to training: [bitrate, frame_size, use_fec, packet_loss_perc]
    for cfg in candidates:
        features = [cfg["bitrate"], cfg["frame_size"], int(cfg["use_fec"]), packet_loss]
        try:
            pred = model.predict([features])[0]
        except Exception:
            pred = 0.0
        if pred > best_score:
            best_score = pred
            best = cfg

    return best


def hybrid_controller(packet_loss, model):
    if packet_loss < 5.0:
        return ml_controller(packet_loss, model)
    else:
        return static_controller(packet_loss)


def run_audio_simulation(audio_path: Path, packet_loss_perc: float, controller_type: str, model=None):
    """Runs processing chain for one controller choice and returns metrics.

    audio_path: Path to original uploaded audio (wav/flac)
    packet_loss_perc: float 0-100
    controller_type: one of 'static','heuristic','ml_adaptive','hybrid'
    model: loaded ML model (joblib) or None
    """
    start = time.time()

    # Ensure source is 16kHz wav for PESQ
    data, sr = sf.read(audio_path)
    data_mono = data[:, 0] if data.ndim > 1 else data
    data_res, sr_res = resample_to_16khz(data_mono, sr)

    temp_input = PROCESSED_AUDIO_DIR / f"temp_input_{int(time.time()*1000)}.wav"
    sf.write(temp_input, data_res, sr_res)

    # Select controller
    if controller_type == 'static':
        cfg = static_controller(packet_loss_perc)
    elif controller_type == 'heuristic':
        cfg = heuristic_controller(packet_loss_perc)
    elif controller_type == 'ml_adaptive':
        cfg = ml_controller(packet_loss_perc, model)
    elif controller_type == 'hybrid':
        cfg = hybrid_controller(packet_loss_perc, model)
    else:
        raise ValueError("Unknown controller type")

    # Run codec pipeline
    processed_file = process_audio_file(
        input_wav_path=temp_input,
        bitrate=cfg['bitrate'],
        frame_size=cfg['frame_size'],
        complexity=cfg['complexity'],
        use_fec=cfg['use_fec'],
        simulated_loss_perc=packet_loss_perc
    )

    if not processed_file:
        # Clean up
        temp_input.unlink(missing_ok=True)
        return None

    # Compute PESQ
    mos = get_audio_quality(temp_input, processed_file)

    end = time.time()

    # Collect metrics
    metrics = {
        "mos": mos,
        "config": cfg,
        "processing_time_s": end - start,
        "processed_file": processed_file
    }

    # Cleanup temp input (keep processed_file for playback if needed)
    temp_input.unlink(missing_ok=True)

    return metrics


def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None
