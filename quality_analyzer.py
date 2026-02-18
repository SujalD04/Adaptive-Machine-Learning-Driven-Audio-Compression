import soundfile as sf
import numpy as np
from pesq import pesq
from pathlib import Path
import warnings

# --- Configuration ---
# Import the same directory settings from our wrapper
from opus_wrapper import CLEAN_AUDIO_DIR, PROCESSED_AUDIO_DIR, process_audio_file

# Define the sample rate we are using. PESQ 'wb' mode requires 16kHz.
REQUIRED_SAMPLE_RATE = 16000
PESQ_MODE = 'wb' # 'wb' = Wideband (for 16kHz)


def get_audio_quality(original_file_path: Path, processed_file_path: Path) -> float | None:
    """
    Compares an original (clean) audio file against a processed (damaged)
    file and returns the PESQ score.

    Args:
        original_file_path: Path to the clean, original .flac or .wav file.
        processed_file_path: Path to the decoded, processed .wav file.

    Returns:
        The PESQ MOS score (float, typically 1.0 to 4.5), or None if an error occurred.
    """
    
    try:
        # 1. Load the original (reference) audio file
        # We use soundfile, which can handle .flac
        ref_data, ref_fs = sf.read(original_file_path, dtype='int16')
        
        # 2. Load the processed (degraded) audio file
        deg_data, deg_fs = sf.read(processed_file_path, dtype='int16')

    except Exception as e:
        print(f"Error reading audio files: {e}")
        return None

    # 3. Validate sample rates
    if ref_fs != REQUIRED_SAMPLE_RATE or deg_fs != REQUIRED_SAMPLE_RATE:
        print(f"Error: Files do not have the required {REQUIRED_SAMPLE_RATE}Hz sample rate.")
        print(f"  Reference: {original_file_path.name} is {ref_fs}Hz")
        print(f"  Degraded: {processed_file_path.name} is {deg_fs}Hz")
        print("  (Did you add '--rate 16000' to opus_wrapper.py?)")
        return None

    # 4. Ensure audio is mono (PESQ expects 1D array)
    # Our VoIP use case is mono.
    if ref_data.ndim > 1:
        ref_data = ref_data[:, 0]
    if deg_data.ndim > 1:
        deg_data = deg_data[:, 0]
        
    # 5. Ensure files are the same length (pad the shorter one)
    # This is common as encoding/decoding can add/remove a few samples
    min_len = min(len(ref_data), len(deg_data))
    ref_data = ref_data[:min_len]
    deg_data = deg_data[:min_len]

    # 6. Calculate PESQ score
    try:
        # Suppress warnings from the PESQ library if it encounters bad audio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = pesq(REQUIRED_SAMPLE_RATE, ref_data, deg_data, PESQ_MODE)
        return score
        
    except Exception as e:
        # This can fail if audio is too corrupted
        print(f"Error during PESQ calculation: {e}")
        # Return a 'floor' score of 1.0 (lowest possible MOS)
        return 1.0


if __name__ == "__main__":
    # This is a test block to see if our full pipeline works
    print("--- Running test on quality_analyzer.py ---")

    # 1. Find a test file
    try:
        original_test_file = next(Path(CLEAN_AUDIO_DIR).rglob("*.flac"))
        print(f"Found original file: {original_test_file}")
    except StopIteration:
        print("Error: Could not find any .flac files in ./clean_audio/LibriSpeech/")
        exit()

    # 2. Define test parameters
    test_params = {
        "input_wav_path": original_test_file,
        "bitrate": 24,           # Lower bitrate
        "frame_size": 20,
        "complexity": 8,
        "use_fec": False,        # NO Forward Error Correction
        "simulated_loss_perc": 5 # 5% packet loss
    }
    
    # 3. Process the file
    print(f"Processing with params: {test_params}")
    processed_file = process_audio_file(**test_params)
    
    if not processed_file:
        print("❌ Error: Audio processing failed. See output above.")
        exit()

    print(f"Processed file created: {processed_file.name}")

    # 4. Analyze the quality
    quality_score = get_audio_quality(original_test_file, processed_file)

    if quality_score:
        print(f"\n✅ Success! Quality analysis complete.")
        print(f"   Score for 5% loss with NO FEC: {quality_score:.4f} MOS")
    else:
        print("\n❌ Error: Quality analysis failed.")
        
    # 5. Run a comparison test
    print("\n--- Running comparison test (FEC vs. No FEC) ---")
    
    # Test 1: No FEC (same as above)
    print(f"Score (No FEC, 5% loss): {quality_score:.4f} MOS")
    
    # Test 2: With FEC
    test_params_fec = test_params.copy()
    test_params_fec["use_fec"] = True
    
    processed_file_fec = process_audio_file(**test_params_fec)
    if processed_file_fec:
        quality_score_fec = get_audio_quality(original_test_file, processed_file_fec)
        print(f"Score (With FEC, 5% loss): {quality_score_fec:.4f} MOS")
        
        if quality_score_fec > quality_score:
            print("\nResult: FEC provided a measurable quality improvement!")
        else:
            print("\nResult: FEC did not provide an improvement in this test.")
            
    # Clean up test files
    processed_file.unlink(missing_ok=True)
    processed_file_fec.unlink(missing_ok=True)