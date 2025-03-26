# main.py
import numpy as np
import pandas as pd
import torch
from config import *
from plots import evaluate_models
from training import kfold_training_and_evaluation, train_on_full_data_and_create_submission
from MatrixVectorizer import MatrixVectorizer
from plots import create_metric_plots, create_radar_chart
import time

def load_and_preprocess_data():
    print("Loading data...")
    lr_data = pd.read_csv(LR_TRAIN_PATH).values  # Shape: (279, 12720)
    hr_data = pd.read_csv(HR_TRAIN_PATH).values  # Shape: (279, 35778)

    # Replace NaNs and negatives with 0/
    lr_data = np.nan_to_num(lr_data, nan=0)
    hr_data = np.nan_to_num(hr_data, nan=0)
    lr_data[lr_data < 0] = 0
    hr_data[hr_data < 0] = 0

    # Convert to tensors
    lr_vectors = torch.tensor(lr_data, dtype=torch.float32)
    hr_vectors = torch.tensor(hr_data, dtype=torch.float32)

    # Anti-vectorize LR data to adjacency matrices
    vectorizer = MatrixVectorizer()
    lr_matrices = [vectorizer.anti_vectorize(vec, 160) for vec in lr_data]
    lr_tensors = torch.tensor(np.array(lr_matrices), dtype=torch.float32)  # Shape: (279, 160, 160)

    print("Data loaded and preprocessed successfully!")
    return lr_tensors, hr_vectors

if __name__ == "__main__":
    lr_tensors, hr_vectors = load_and_preprocess_data()
    # kfold_training_and_evaluation(lr_tensors, hr_vectors)
    # ===========================
    # Step 5: Run Evaluation
    # ===========================
    fold_metrics, fold_means, fold_stds, overall_means, overall_stds = evaluate_models()
    train_on_full_data_and_create_submission(lr_tensors, hr_vectors)
    print("\n🎉 Pipeline complete! 🎉")
    print("1. Cross-validation models saved as 'model_fold_X.pth'")
    print("2. Best cross-validation model saved as 'best_cv_model.pth'")
    print("3. Final model trained on all data saved as 'final_model.pth'")
    print("4. Test predictions saved as 'submission.csv'")
    print("5. Evaluation metrics visualized in 'evaluation_metrics_by_fold.png'")
    print("6. Radar chart saved as 'metrics_radar_chart.png'")