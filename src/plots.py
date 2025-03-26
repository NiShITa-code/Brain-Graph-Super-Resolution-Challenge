# plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import random  
import pandas as pd
from MatrixVectorizer import MatrixVectorizer
from model import ImprovedGraphSRModel
from config import *
from training import create_batched_data
import networkx as nx
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import KFold

# Set plot style
plt.style.use('ggplot')
sns.set_context("talk")



def compute_small_worldness(G):
    """
    Compute small-worldness (σ) as (C/Crand) / (L/Lrand) for weighted graphs,
    preserving the weight distribution in the random comparison graph.
    """
    # If graph is disconnected, use largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    try:
        # Calculate metrics for original graph
        C = nx.average_clustering(G, weight="weight")
        L = nx.average_shortest_path_length(G, weight="weight")

        # Create random graph with same number of nodes and edges
        n = G.number_of_nodes()
        m = G.number_of_edges()

        # Extract weights from original graph
        weights = [G[u][v]["weight"] for u, v in G.edges()]

        # Generate Erdos-Renyi random graph with same number of nodes and edges
        p = (2.0 * m) / (n * (n - 1))
        rand_G = nx.erdos_renyi_graph(n, p)

        # Ensure the random graph has at least the same number of edges
        while rand_G.number_of_edges() < m:
            # Add edges until we have the same number
            potential_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not rand_G.has_edge(i, j)]
            if not potential_edges:
                break
            u, v = random.choice(potential_edges)
            rand_G.add_edge(u, v)

        # If we have too many edges, remove some
        while rand_G.number_of_edges() > m:
            # Remove a random edge
            u, v = random.choice(list(rand_G.edges()))
            rand_G.remove_edge(u, v)

        # Now randomly assign the original weights to the edges in rand_G
        random.shuffle(weights)
        for (u, v), weight in zip(rand_G.edges(), weights):
            rand_G[u][v]["weight"] = weight

        # Calculate metrics for random graph
        C_rand = nx.average_clustering(rand_G, weight="weight")
        if nx.is_connected(rand_G):
            L_rand = nx.average_shortest_path_length(rand_G, weight="weight")
        else:
            largest_cc_rand = max(nx.connected_components(rand_G), key=len)
            rand_G_connected = rand_G.subgraph(largest_cc_rand).copy()
            L_rand = nx.average_shortest_path_length(rand_G_connected, weight="weight")

        # Calculate small-worldness
        sigma = (C / C_rand) / (L / L_rand)
        return sigma
    except Exception as e:
        print(f"Error calculating small-worldness: {e}")
        return 0

# ===========================
# Evaluation Metrics Functions
# ===========================
def calculate_metrics(pred_matrices, gt_matrices):
    """Calculate evaluation metrics for brain graph super-resolution."""
    num_samples = len(pred_matrices)

    # Lists to store individual metrics
    mae_bc = []
    mae_ec = []
    mae_pc = []
    ge_diffs = []
    sw_diffs = []  # small-worldness differences

    # Lists to store vectorized matrices
    pred_1d_list = []
    gt_1d_list = []

    for i in range(num_samples):
        # Convert matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i], edge_attr="weight")

        try:
            # Compute centrality measures
            pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
            pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight", max_iter=1000)
            pred_pc = nx.pagerank(pred_graph, weight="weight")

            gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
            gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight", max_iter=1000)
            gt_pc = nx.pagerank(gt_graph, weight="weight")

            # Convert dictionaries to lists
            pred_bc_values = list(pred_bc.values())
            pred_ec_values = list(pred_ec.values())
            pred_pc_values = list(pred_pc.values())

            gt_bc_values = list(gt_bc.values())
            gt_ec_values = list(gt_ec.values())
            gt_pc_values = list(gt_pc.values())

            # Compute MAEs for centrality measures
            mae_bc.append(mean_absolute_error(gt_bc_values, pred_bc_values))
            mae_ec.append(mean_absolute_error(gt_ec_values, pred_ec_values))
            mae_pc.append(mean_absolute_error(gt_pc_values, pred_pc_values))

            # Global Efficiency differences
            pred_ge = nx.global_efficiency(pred_graph)
            gt_ge = nx.global_efficiency(gt_graph)
            ge_diffs.append(abs(pred_ge - gt_ge))

            # Small-worldness differences
            pred_sw = compute_small_worldness(pred_graph)
            gt_sw = compute_small_worldness(gt_graph)
            sw_diffs.append(abs(pred_sw - gt_sw))

        except Exception as e:
            print(f"Error in centrality calculation for sample {i}: {e}")
            mae_bc.append(np.nan)
            mae_ec.append(np.nan)
            mae_pc.append(np.nan)
            ge_diffs.append(np.nan)
            sw_diffs.append(np.nan)

        # Vectorize matrices for global metrics
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    mae = mean_absolute_error(gt_1d, pred_1d)
    pcc = pearsonr(gt_1d, pred_1d)[0]
    js_dis = jensenshannon(gt_1d, pred_1d)

    avg_mae_bc = np.nanmean(mae_bc)
    avg_mae_ec = np.nanmean(mae_ec)
    avg_mae_pc = np.nanmean(mae_pc)
    avg_ge_diff = np.nanmean(ge_diffs)
    avg_sw_diff = np.nanmean(sw_diffs)

    std_mae_bc = np.nanstd(mae_bc)
    std_mae_ec = np.nanstd(mae_ec)
    std_mae_pc = np.nanstd(mae_pc)
    std_ge_diff = np.nanstd(ge_diffs)
    std_sw_diff = np.nanstd(sw_diffs)

    individual_metrics = {
        'MAE_BC': mae_bc,
        'MAE_EC': mae_ec,
        'MAE_PC': mae_pc,
        'GE': ge_diffs,
        'SW': sw_diffs   # small-worldness differences
    }

    mean_metrics = {
        'MAE': mae,
        'PCC': pcc,
        'JSD': js_dis,
        'MAE_PC': avg_mae_pc,
        'MAE_EC': avg_mae_ec,
        'MAE_BC': avg_mae_bc,
        'GE': avg_ge_diff,
        'SW': avg_sw_diff
    }

    std_metrics = {
        'MAE': 0,
        'PCC': 0,
        'JSD': 0,
        'MAE_PC': std_mae_pc,
        'MAE_EC': std_mae_ec,
        'MAE_BC': std_mae_bc,
        'GE': std_ge_diff,
        'SW': std_sw_diff
    }

    return individual_metrics, mean_metrics, std_metrics

# ===========================
# Visualization Functions
# ===========================
def create_radar_chart(overall_means, overall_stds):
    """Create a radar chart for the average metrics."""
    metrics = ['MAE', 'PCC', 'JSD', 'MAE_PC', 'MAE_EC', 'MAE_BC', 'GE', 'SW']
    normalized_values = []
    for m in metrics:
        value = overall_means[m]
        # For PCC and SW, higher is better. For others, lower is better.
        if m in ['PCC', 'SW']:
            normalized_values.append(value)
        else:
            max_possible = 1.0
            normalized_values.append(1.0 - min(value/max_possible, 1.0))

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    normalized_values += normalized_values[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics, size=12)

    ax.plot(angles, normalized_values, linewidth=2, linestyle='solid')
    ax.fill(angles, normalized_values, alpha=0.25)

    plt.title('Model Performance Radar Chart', size=15)
    plt.tight_layout()
    plt.savefig('performance_radar_chart.png', dpi=300, bbox_inches='tight')
    print("Radar chart saved to 'performance_radar_chart.png'")

def create_metric_plots(fold_means, fold_stds, overall_means, overall_stds):
    """Create plots for each fold's metrics and an average plot with error bars."""
    metrics = ['MAE', 'PCC', 'JSD', 'MAE_PC', 'MAE_EC', 'MAE_BC', 'GE', 'SW']
    metric_full_names = {
        'MAE': 'Mean Absolute Error (MAE)',
        'PCC': 'Pearson Correlation Coefficient (PCC)',
        'JSD': 'Jensen-Shannon Divergence (JSD)',
        'MAE_PC': 'PageRank Centrality MAE (MAE_PC)',
        'MAE_EC': 'Eigenvector Centrality MAE (MAE_EC)',
        'MAE_BC': 'Betweenness Centrality MAE (MAE_BC)',
        'GE': 'Global Efficiency Difference',
        'SW': 'Small-Worldness Difference'
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    metric_colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

    # Plot for each fold (assume 3 folds)
    for fold in range(min(3, len(fold_means))):
        ax = axes[fold]
        x_pos = np.arange(len(metrics))
        values = [fold_means[fold][metric] for metric in metrics]
        ax.bar(x_pos, values, alpha=0.7, color=metric_colors)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Fold {fold+1} Metrics')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        max_val = max([max([fold_means[i][m] for m in metrics]) for i in range(len(fold_means))])
        ax.set_ylim(0, max_val * 1.2)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Plot average metrics across folds
    ax = axes[3]
    x_pos = np.arange(len(metrics))
    avg_values = [overall_means[metric] for metric in metrics]
    err_values = [overall_stds[metric] for metric in metrics]
    ax.bar(x_pos, avg_values, yerr=err_values, capsize=10, alpha=0.7, color=metric_colors, ecolor='black')
    for i, (v, e) in enumerate(zip(avg_values, err_values)):
        ax.text(i, v + e + 0.02, f'{v:.3f}±{e:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    ax.set_ylabel('Metric Value')
    ax.set_title('Average Metrics Across All Folds (with Standard Deviations)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    max_val_with_err = max([overall_means[m] + overall_stds[m] for m in metrics])
    ax.set_ylim(0, max_val_with_err * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    legend_handles = [plt.Rectangle((0,0),1,1, color=metric_colors[i]) for i in range(len(metrics))]
    fig.legend(legend_handles, [metric_full_names[m] for m in metrics],
               loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=4, fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle('Evaluation Metrics by Fold', fontsize=16, y=0.99)
    plt.savefig('evaluation_metrics_by_fold.png', dpi=300, bbox_inches='tight')
    print("Evaluation plots saved to 'evaluation_metrics_by_fold.png'")
    create_radar_chart(overall_means, overall_stds)

# ===========================
# Main Evaluation Function
# ===========================
def evaluate_models():
    print("=== Model Evaluation with Visualization ===")
    print("Starting model evaluation...")
    start_time = time.time()
    print("Loading validation data...")
    hr_data = pd.read_csv(HR_TRAIN_PATH).values
    lr_data = pd.read_csv(LR_TRAIN_PATH).values
    hr_data = np.nan_to_num(hr_data, nan=0)
    lr_data = np.nan_to_num(lr_data, nan=0)
    hr_data[hr_data < 0] = 0
    lr_data[lr_data < 0] = 0

    vectorizer = MatrixVectorizer()
    lr_matrices = [vectorizer.anti_vectorize(vec, 160) for vec in lr_data]
    lr_tensors = torch.tensor(np.array(lr_matrices), dtype=torch.float32)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_indices = list(kf.split(lr_tensors))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ImprovedGraphSRModel(
        in_channels=161, hidden_channels=128, embedding_dim=64, output_dim=35778, dropout=0.5
    ).to(device)

    all_fold_metrics = []
    all_fold_means = []
    all_fold_stds = []

    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        fold_num = fold + 1
        print(f"\n=== Evaluating Fold {fold_num} ===")
        try:
            state_dict = torch.load(f"best_model_fold_{fold_num}.pth", map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model weights for fold {fold_num} loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights for fold {fold_num}: {e}")
            continue

        model.eval()
        with torch.no_grad():
            val_idx_list = val_idx.tolist()
            val_x_batch, val_edge_index_batch, val_batch = create_batched_data(val_idx_list, lr_tensors, device)
            val_pred_vectors, _ = model(val_x_batch, val_edge_index_batch, val_batch)
            val_pred_vectors = val_pred_vectors.cpu().numpy()
            val_hr_vectors = hr_data[val_idx_list]

            pred_matrices = []
            gt_matrices = []
            for i in range(len(val_idx_list)):
                pred_matrix = vectorizer.anti_vectorize(val_pred_vectors[i], 160)
                gt_matrix = vectorizer.anti_vectorize(val_hr_vectors[i], 160)
                pred_matrices.append(pred_matrix)
                gt_matrices.append(gt_matrix)

            fold_metrics, fold_means, fold_stds = calculate_metrics(pred_matrices, gt_matrices)
            all_fold_metrics.append(fold_metrics)
            all_fold_means.append(fold_means)
            all_fold_stds.append(fold_stds)

            print(f"Fold {fold_num} Metrics:")
            for metric, value in fold_means.items():
                print(f"  {metric}: {value:.4f} ± {fold_stds[metric]:.4f}")

    overall_means = {}
    overall_stds = {}
    metric_keys = ['MAE', 'PCC', 'JSD', 'MAE_PC', 'MAE_EC', 'MAE_BC', 'GE', 'SW']
    for metric in metric_keys:
        metric_means = [fold_means[metric] for fold_means in all_fold_means]
        overall_means[metric] = np.mean(metric_means)
        overall_stds[metric] = np.std(metric_means)

    print("\n=== Overall Results (Averaged Across Folds) ===")
    for metric, value in overall_means.items():
        print(f"{metric}: {value:.4f} ± {overall_stds[metric]:.4f}")

    create_metric_plots(all_fold_means, all_fold_stds, overall_means, overall_stds)
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")
    print("\nComplete! Check the output directory for evaluation plots.")
    return all_fold_metrics, all_fold_means, all_fold_stds, overall_means, overall_stds
