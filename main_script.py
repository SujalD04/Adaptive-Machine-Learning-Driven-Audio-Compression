import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import atexit
import warnings

# Import our helper functions from the other files
from opus_wrapper import process_audio_file, CLEAN_AUDIO_DIR, PROCESSED_AUDIO_DIR, BASE_PROJECT_DIR
from quality_analyzer import get_audio_quality

# ----------------------------------------------------------------------------
# 💡 PARAMETER SWEEP (FOR 10k SAMPLES) 💡
# ----------------------------------------------------------------------------

# Set this to 20 to get ~10,000 samples
MAX_FILES_TO_PROCESS = 20

# List of bitrates (kbps)
BITRATES = [8, 12, 16, 24, 32, 48]

# List of frame sizes (ms)
FRAME_SIZES = [20, 40, 60]

# Encoder complexity (0-10).
COMPLEXITIES = [5, 9]

# FEC status (True=On, False=Off)
FEC_STATUSES = [True, False]

# Simulated Packet Loss Percentages
PACKET_LOSSES = [0, 1, 2, 3, 5, 7, 10]

# --- Global Pool for cleanup ---
# We'll create a global pool to be able to close it on exit
pool = None

def cleanup_pool():
    """Ensure the process pool is terminated when the script exits."""
    global pool
    if pool:
        print("Terminating process pool...")
        pool.terminate()
        pool.join()
        print("Pool terminated.")

# Register the cleanup function to be called on script exit
atexit.register(cleanup_pool)

def run_single_combination(params: dict) -> dict | None:
    """
    This is our "worker" function. It processes ONE combination of parameters.
    It's designed to be run in a separate process.
    """
    # Suppress PESQ warnings within the worker process
    warnings.simplefilter("ignore")
    
    try:
        # --- Step 1: Process the audio ---
        processed_file = process_audio_file(
            input_wav_path=params["original_file"],
            bitrate=params["bitrate"],
            frame_size=params["frame_size"],
            complexity=params["complexity"],
            use_fec=params["use_fec"],
            simulated_loss_perc=params["packet_loss_perc"]
        )

        if not processed_file:
            # This can happen if opusenc fails
            return None

        # --- Step 2: Analyze the quality ---
        score = get_audio_quality(params["original_file"], processed_file)

        if not score:
            # This can happen if pesq fails
            processed_file.unlink(missing_ok=True) # Clean up
            return None

        # --- Step 3: Create the result ---
        result_data = {
            "original_file": params["original_file"].name,
            "bitrate": params["bitrate"],
            "frame_size": params["frame_size"],
            "complexity": params["complexity"],
            "use_fec": params["use_fec"],
            "packet_loss_perc": params["packet_loss_perc"],
            "pesq_mos_score": score
        }

        # --- Step 4: Clean up the file ---
        processed_file.unlink(missing_ok=True)
        
        return result_data
        
    except Exception as e:
        # Catch any other unexpected errors in the worker
        # print(f"Error in worker process: {e}") # Can be noisy
        return None


def run_experiment_parallel():
    """
    Runs the full experiment in parallel using all available CPU cores.
    """
    global pool # Access the global pool
    print("--- Starting Phase 1: Data Generation (Parallel) ---")

    # 1. Find all our clean audio files
    audio_files = list(Path(CLEAN_AUDIO_DIR).rglob("*.flac"))
    if not audio_files:
        print(f"Error: No .flac files found in {CLEAN_AUDIO_DIR}")
        return

    files_to_process = audio_files[:MAX_FILES_TO_PROCESS]
    print(f"Found {len(audio_files)} total files. Processing {len(files_to_process)}.")

    # 2. Create the "Job List" - all 10,080 parameter combinations
    print("Generating job list...")
    all_job_params = []
    for original_file in files_to_process:
        for bitrate in BITRATES:
            for frame_size in FRAME_SIZES:
                for complexity in COMPLEXITIES:
                    for use_fec in FEC_STATUSES:
                        for loss_perc in PACKET_LOSSES:
                            params = {
                                "original_file": original_file,
                                "bitrate": bitrate,
                                "frame_size": frame_size,
                                "complexity": complexity,
                                "use_fec": use_fec,
                                "packet_loss_perc": loss_perc
                            }
                            all_job_params.append(params)

    total_runs = len(all_job_params)
    print(f"Total combinations to process: {total_runs}")

    # 3. Run the "Job List" in Parallel
    num_cores = multiprocessing.cpu_count()
    print(f"Starting process pool with {num_cores} cores (you have 12)...")

    results_list = []
    
    try:
        # Create the pool
        pool = multiprocessing.Pool(processes=num_cores)
        
        # Use pool.imap_unordered to get results as they finish.
        # This gives us a much more responsive progress bar.
        jobs = pool.imap_unordered(run_single_combination, all_job_params)
        
        # Wrap the 'jobs' iterable with tqdm to create the progress bar
        for result in tqdm(jobs, total=total_runs, unit="run"):
            if result is not None:
                results_list.append(result)
                
        # We are done, close the pool
        pool.close()
        pool.join()
        pool = None # Clear the global
        
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt! Terminating workers...")
        # cleanup_pool() is already registered with atexit, so it will run
        return
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        # cleanup_pool() will run
        return

    print(f"\n✅ --- Experiment Complete! ---")

    # 4. Save all results to a CSV file
    if results_list:
        output_csv_path = BASE_PROJECT_DIR / "opus_dataset.csv"
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv_path, index=False)
        print(f"Successfully generated dataset with {len(df)} rows.")
        print(f"Dataset saved to: {output_csv_path}")
    else:
        print("No results were generated. Please check for errors.")


if __name__ == "__main__":
    # This is necessary for multiprocessing to work correctly on some systems
    multiprocessing.freeze_support() 
    run_experiment_parallel()