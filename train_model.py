import pandas as pd
from pathlib import Path
import joblib  # We use joblib to save the model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# --- Configuration ---
BASE_PROJECT_DIR = Path.home() / "adaptive_opus"
DATASET_PATH = BASE_PROJECT_DIR / "opus_dataset.csv"
MODEL_OUTPUT_PATH = BASE_PROJECT_DIR / "qoe_model.joblib"


def train_model():
    """
    Loads the dataset, trains a Random Forest model, and saves it.
    """
    print("--- Starting Phase 2: Model Training ---")

    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please run main_script.py first to generate the dataset.")
        return
    
    print(f"Loaded dataset with {len(df)} rows.")

    # 2. Pre-process Data (Feature Engineering)
    
    # Create a copy to avoid warnings
    df_processed = df.copy()
    
    # Convert 'use_fec' (True/False) to a number (1/0)
    # This is required for the ML model
    df_processed['use_fec'] = df_processed['use_fec'].astype(int)
    
    # We can drop columns we don't need for the model
    df_processed = df_processed.drop(columns=['original_file', 'complexity'])

    # 3. Define Features (X) and Target (y)
    # The model learns to predict 'y' using the features in 'X'
    feature_names = [
        'bitrate', 
        'frame_size', 
        'use_fec', 
        'packet_loss_perc'
    ]
    target_name = 'pesq_mos_score'
    
    X = df_processed[feature_names]
    y = df_processed[target_name]

    # 4. Split Data into Training and Testing sets
    # 80% for training, 20% for testing
    # random_state ensures we get the same "random" split every time
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Splitting data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 5. Initialize and Train the Model
    print("Training Random Forest Regressor...")
    
    # n_estimators = number of "trees" in the forest. 100 is a good default.
    # random_state=42 makes the model's "randomness" reproducible
    # n_jobs=-1 uses all available CPU cores to speed up training
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # The 'fit' command is where the model does all its learning
    model.fit(X_train, y_train)

    print("Training complete.")

    # 6. Evaluate the Model
    print("--- Model Evaluation ---")
    
    # Use the model to predict scores on the "unseen" test data
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R²) Score:     {r2:.4f}")
    print("  (MSE should be as close to 0 as possible)")
    print("  (R² should be as close to 1.0 as possible)")
    
    # 7. Save the Trained Model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\n✅ --- Model Saved! ---")
    print(f"The trained model is saved to: {MODEL_OUTPUT_PATH}")
    print("This file is your 'ML-Based Adaptive Controller'.")


if __name__ == "__main__":
    train_model()