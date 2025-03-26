# training_and_evaluation.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch_geometric.utils import dense_to_sparse, degree
from torch_scatter import scatter_mean
import numpy as np
import pandas as pd
import time
import psutil
import os
from config import *
from model import ImprovedGraphSRModel
from MatrixVectorizer import MatrixVectorizer
 
def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB

def topological_loss(node_embeddings, edge_index):
    row, col = edge_index
    similarity = (node_embeddings[row] * node_embeddings[col]).sum(dim=1)
    pred_edges = torch.sigmoid(similarity)
    target = torch.ones_like(pred_edges)
    loss = nn.BCELoss()(pred_edges, target)
    return loss

def create_batched_data(indices, lr_tensors, device):
    x_list, edge_index_list, batch_list = [], [], []
    for local_idx, global_idx in enumerate(indices):
        adj = lr_tensors[global_idx].to(device)
        edge_index_i = dense_to_sparse(adj)[0]
        deg_i = degree(edge_index_i[0], num_nodes=160).unsqueeze(1)
        x_i = torch.cat([adj, deg_i], dim=1)
        x_list.append(x_i)
        edge_index_i_offset = edge_index_i + local_idx * 160
        edge_index_list.append(edge_index_i_offset)
        batch_list.append(torch.full((160,), local_idx, dtype=torch.long, device=device))

    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    batch = torch.cat(batch_list, dim=0)
    return x_batch, edge_index_batch, batch

def kfold_training_and_evaluation(lr_tensors, hr_vectors):
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_seed)
    total_training_time = 0
    fold_times = []
    peak_ram_usage = get_ram_usage()

    for fold, (train_idx, val_idx) in enumerate(kf.split(lr_tensors)):
        print(f"\n=== Fold {fold + 1} ===")
        fold_start_time = time.time()

        model = ImprovedGraphSRModel(IN_CHANNELS, HIDDEN_CHANNELS, EMBEDDING_DIM, OUTPUT_DIM, DROPOUT, NUM_HEADS).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.L1Loss()

        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()

        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            np.random.shuffle(train_idx)

            for batch_start in range(0, len(train_idx), BATCH_SIZE):
                batch_global_idx = train_idx[batch_start:batch_start + BATCH_SIZE]
                x_batch, edge_index_batch, batch = create_batched_data(batch_global_idx, lr_tensors, device)
                hr_vectors_batch = hr_vectors[batch_global_idx].to(device)

                optimizer.zero_grad()
                pred_vectors, node_embeddings = model(x_batch, edge_index_batch, batch)
                primary_loss = loss_fn(pred_vectors, hr_vectors_batch)
                topo_loss = topological_loss(node_embeddings, edge_index_batch)
                total_loss = primary_loss + LAMBDA_TOPO * topo_loss
                total_loss.backward()
                optimizer.step()

                current_ram = get_ram_usage()
                peak_ram_usage = max(peak_ram_usage, current_ram)
                epoch_loss += total_loss.item()
                batch_count += 1

            avg_epoch_loss = epoch_loss / batch_count

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_x_batch, val_edge_index_batch, val_batch = create_batched_data(val_idx, lr_tensors, device)
                    val_hr_vectors = hr_vectors[val_idx].to(device)
                    val_pred_vectors,_ = model(val_x_batch, val_edge_index_batch, val_batch)
                    val_loss = loss_fn(val_pred_vectors, val_hr_vectors)
                    epoch_time = time.time() - epoch_start_time
                    print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss}, Validation Loss: {val_loss.item():.4f}, RAM: {get_ram_usage():.2f} GB")

        model.eval()
        with torch.no_grad():
            val_x_batch, val_edge_index_batch, val_batch = create_batched_data(val_idx, lr_tensors, device)
            pred_vectors, _ = model(val_x_batch, val_edge_index_batch, val_batch)
            pred_vectors = pred_vectors.cpu().numpy()

            val_hr_vectors_np = hr_vectors[val_idx].numpy()
            mae = np.mean(np.abs(pred_vectors - val_hr_vectors_np))
            print(f"Fold {fold + 1} Validation MAE: {mae:.4f}")

            pred_df = pd.DataFrame(pred_vectors)
            pred_df.to_csv(f"predictions_fold__{fold + 1}.csv", index=False)

        torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")

        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        total_training_time += fold_time

        print(f"Model saved for fold {fold + 1}")
        print(f"Fold {fold + 1} training time: {fold_time:.2f} seconds ({fold_time/60:.2f} minutes)")
        print(f"Current RAM usage: {get_ram_usage():.2f} GB")

    print("\n===== Training Summary =====")
    print(f"Total 3F-CV training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Average time per fold: {(total_training_time/len(fold_times)):.2f} seconds")
    print(f"Peak RAM usage: {peak_ram_usage:.2f} GB")
    print("Training and validation complete. Predictions saved successfully!")



def train_on_full_data_and_create_submission(lr_tensors, hr_vectors):
    num_epochs = 35
    lambda_topo = 0.1

    # Initialize model
    full_model = ImprovedGraphSRModel(
        in_channels=161, hidden_channels=128, embedding_dim=64, output_dim=35778, dropout=0.6
    ).to(device)
    optimizer = optim.AdamW(full_model.parameters(), lr=0.001)
    loss_fn = nn.L1Loss()

    # Define full training indices
    train_idx = list(range(len(lr_tensors)))  # All 279 samples

    # Training loop
    for epoch in range(num_epochs):
        full_model.train()
        np.random.shuffle(train_idx)  # Shuffle indices each epoch
        epoch_loss = 0.0
        batch_count = 0

        for batch_start in range(0, len(train_idx), BATCH_SIZE):
            batch_global_idx = train_idx[batch_start:batch_start + BATCH_SIZE]
            x_batch, edge_index_batch, batch = create_batched_data(batch_global_idx, lr_tensors, device)
            hr_vectors_batch = hr_vectors[batch_global_idx].to(device)
            
            optimizer.zero_grad()
            pred_vectors, node_embeddings = full_model(x_batch, edge_index_batch, batch)
            primary_loss = loss_fn(pred_vectors, hr_vectors_batch)
            topo_loss = topological_loss(node_embeddings, edge_index_batch)
            total_loss = primary_loss + lambda_topo * topo_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count

        # Print training loss every 10 epochs
        # if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss:.4f}")

    # Save the final model
    torch.save(full_model.state_dict(), "final_model.pth")
    print("Final model saved successfully!")

    # ===========================
    # Step 8: Generate Test Predictions
    # ===========================
    print("\n================================")
    print("PHASE 3: GENERATING TEST PREDICTIONS")
    print("================================")

    # Load test data
    print("Loading test data...")
    test_lr_data = pd.read_csv(LR_TEST_PATH).values  # Shape (112, 12720)

    # Preprocess test data
    test_lr_data = np.nan_to_num(test_lr_data, nan=0)  # Replace NaNs with 0
    test_lr_data[test_lr_data < 0] = 0  # Replace negative values with 0

    # Convert vectorized data to adjacency matrices
    vectorizer = MatrixVectorizer()
    test_lr_matrices = [vectorizer.anti_vectorize(vec, 160) for vec in test_lr_data]
    test_lr_tensors = torch.tensor(np.array(test_lr_matrices), dtype=torch.float32)

    print(f"Test data loaded and preprocessed. Shape: {test_lr_tensors.shape}")

    # Prepare test data for prediction
    def prepare_test_data(test_lr_tensors, device):
        num_test_samples = len(test_lr_tensors)
        test_x_list, test_edge_index_list, test_batch_list = [], [], []

        for i in range(num_test_samples):
            adj = test_lr_tensors[i].to(device)
            edge_index_i = dense_to_sparse(adj)[0]
            deg_i = degree(edge_index_i[0], num_nodes=160).unsqueeze(1)
            x_i = torch.cat([adj, deg_i], dim=1)  # Shape (160, 161)
            test_x_list.append(x_i)

            # Add correct offset for batching
            edge_index_i_offset = edge_index_i + i * 160
            test_edge_index_list.append(edge_index_i_offset)

            batch_i = torch.full((160,), i, dtype=torch.long, device=device)
            test_batch_list.append(batch_i)

        test_x_batch = torch.cat(test_x_list, dim=0)
        test_edge_index_batch = torch.cat(test_edge_index_list, dim=1)
        test_batch = torch.cat(test_batch_list, dim=0)

        return test_x_batch, test_edge_index_batch, test_batch

    # Prepare test data
    test_x_batch, test_edge_index_batch, test_batch = prepare_test_data(test_lr_tensors, device)

    # Print shapes for verification
    print(f"test_x_batch shape: {test_x_batch.shape}")
    print(f"test_edge_index_batch shape: {test_edge_index_batch.shape}")
    print(f"test_batch shape: {test_batch.shape}")

    # Generate predictions using the final model
    full_model.eval()
    with torch.no_grad():
        test_pred_vectors,_ = full_model(test_x_batch, test_edge_index_batch, test_batch)
        test_pred_vectors = test_pred_vectors.cpu().numpy()  # Shape (112, 35778)

    # Create submission file
    print("Creating submission file...")
    flattened_predictions = test_pred_vectors.flatten()  # Shape (4007136,)
    ids = np.arange(1, len(flattened_predictions) + 1)  # IDs from 1 to 4007136

    # Create DataFrame for submission
    submission_df = pd.DataFrame({
        "ID": ids,
        "Predicted": flattened_predictions
    })

    # Save submission file
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")
