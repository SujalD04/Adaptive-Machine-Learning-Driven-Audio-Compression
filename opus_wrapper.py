import subprocess
import os
from pathlib import Path

# --- Configuration ---
# These paths are inside your WSL2 environment.
# Make sure your project folder is in your home directory.
BASE_PROJECT_DIR = Path.home() / "adaptive_opus"
CLEAN_AUDIO_DIR = BASE_PROJECT_DIR / "clean_audio"
PROCESSED_AUDIO_DIR = BASE_PROJECT_DIR / "processed_audio"

# Ensure the processed audio directory exists
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def process_audio_file(
    input_wav_path: Path,
    bitrate: int,
    frame_size: int,
    complexity: int,
    use_fec: bool,
    simulated_loss_perc: float
) -> Path | None:
    """
    Runs a single audio file through the full Opus encode/decode pipeline
    to simulate network conditions and codec settings.

    Args:
        input_wav_path: Path to the clean (original) .wav file.
        bitrate: Target bitrate in kbps (e.g., 32).
        frame_size: Frame size in ms (e.g., 20, 40, 60).
        complexity: Encoder complexity (0-10, 10 is best).
        use_fec: Boolean. If True, tells the encoder to expect loss (enables FEC).
        simulated_loss_perc: The percentage of packets to *actually* drop (0-100).

    Returns:
        The path to the final decoded (and damaged) .wav file, or None if an error occurred.
    """
    
    # 1. Create a unique filename for this processing run
    # Example: original_file_b32_f20_c10_fec_l5.wav
    fec_str = "fec" if use_fec else "nofec"
    loss_str = str(simulated_loss_perc).replace('.', 'p')
    
    output_filename_base = (
        f"{input_wav_path.stem}_"
        f"b{bitrate}_f{frame_size}_c{complexity}_"
        f"{fec_str}_l{loss_str}"
    )
    
    # Define the intermediate and final file paths
    # We use .opus for the intermediate compressed file
    intermediate_opus_file = PROCESSED_AUDIO_DIR / f"{output_filename_base}.opus"
    # And .wav for the final, decoded (damaged) file
    final_wav_file = PROCESSED_AUDIO_DIR / f"{output_filename_base}.wav"

    # --- 2. Step 2: ENCODE (Clean .wav -> Compressed .opus) ---
    
    # Set the --expect-loss flag if FEC is requested.
    # This tells the encoder to add redundant data (FEC) to the stream.
    fec_flag = ["--expect-loss", "5"] if use_fec else [] # '5' is a reasonable default guess for FEC
    
    encode_command = [
        "opusenc",
        "--bitrate", str(bitrate),
        "--framesize", str(frame_size),
        "--comp", str(complexity),
        *fec_flag,
        str(input_wav_path),
        str(intermediate_opus_file)
    ]

    try:
        # Run the command. check=True means it will raise an error if opusenc fails
        subprocess.run(encode_command, check=True, capture_output=True, text=True)
    
    except subprocess.CalledProcessError as e:
        print(f"Error during ENCODING. File: {input_wav_path.name}")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr: {e.stderr}")
        return None

    # --- 3. Step 3: DECODE (Compressed .opus -> Damaged .wav) ---
    
    # Set the --packet-loss flag.
    # This tells the decoder to *simulate* random packet loss.
    
    # ***** NEW (AND CRITICAL) *****
    # We force the output sample rate to 16000 Hz.
    # The original LibriSpeech files are 16kHz, and the 'pesq'
    # library requires either 8kHz or 16kHz. This ensures both
    # original and processed files match.
    
    decode_command = [
        "opusdec",
        "--rate", "16000",  # <--- ADD THIS LINE
        "--packet-loss", str(simulated_loss_perc),
        str(intermediate_opus_file),
        str(final_wav_file)
    ]
    
    try:
        subprocess.run(decode_command, check=True, capture_output=True, text=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during DECODING. File: {intermediate_opus_file.name}")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Stderr: {e.stderr}")
        # Clean up the intermediate file if decoding fails
        intermediate_opus_file.unlink(missing_ok=True)
        return None

    # --- 4. Step 4: Clean up intermediate file ---
    intermediate_opus_file.unlink(missing_ok=True)
    
    # Return the path to the final, processed .wav file
    return final_wav_file


if __name__ == "__main__":
    # This is a test block to see if our function works.
    print("--- Running test on opus_wrapper.py ---")
    
    # 1. Find the first .flac file in our clean_audio directory to use for testing
    # LibriSpeech uses .flac, but opus-tools can read them directly!
    try:
        test_file = next(Path(CLEAN_AUDIO_DIR).rglob("*.flac"))
        print(f"Found test file: {test_file}")
    except StopIteration:
        print("Error: Could not find any .flac files in ./clean_audio/LibriSpeech/")
        print("Please make sure you ran Step 4 from the setup instructions.")
        exit()

    # 2. Define test parameters
    test_params = {
        "input_wav_path": test_file,
        "bitrate": 32,
        "frame_size": 20,
        "complexity": 10,
        "use_fec": True,         # Test with FEC enabled
        "simulated_loss_perc": 5 # Test with 5% packet loss
    }

    # 3. Run the processing function
    print(f"Processing with params: {test_params}")
    output_file = process_audio_file(**test_params)

    if output_file and output_file.exists():
        print(f"\n✅ Success! Test complete.")
        print(f"   Original file: {test_file.name}")
        print(f"   Processed file: {output_file.name}")
        print(f"   Saved to: {output_file}")
    else:
        print("\n❌ Error: Test failed. See output above.")