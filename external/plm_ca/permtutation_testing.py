import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from explainable_medical_coding.eval.metrics_permutations import (
    MetricCollection,
    F1Score,
    Precision_K,
    ExactMatchRatio,
    MeanAveragePrecision,
    AUC2,  # Note: AUC is quite memory intensive and may crash with RAM <32GB.
           # Passing a thresholds parameter (see torchmetrics documentation) may help.
    # Add other metrics if needed
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer


def evaluate_predictions(df: pd.DataFrame, model_threshold: float) -> dict:
    """
    Evaluate predictions by computing various metrics.

    Args:
        df (pd.DataFrame): DataFrame containing target and prediction columns.
        model_threshold (float): Threshold value for computing metrics.

    Returns:
        dict: A dictionary of computed metrics.
    """
    excluded_cols = {'_id', 'target'}
    class_cols = [col for col in df.columns if col not in excluded_cols]
    
    if not class_cols:
        raise ValueError("No class probability columns found.")
    
    # Map each class label to an index
    class_to_index = {label: idx for idx, label in enumerate(class_cols)}
    num_classes = len(class_cols)

    # Vectorized Multi-hot Encoding for Targets
    targets_array = np.zeros((len(df), num_classes), dtype=np.float32)
    for i, labels in enumerate(df['target']):
        indices = [class_to_index[label] for label in labels if label in class_to_index]
        targets_array[i, indices] = 1.0

    targets = torch.tensor(targets_array, dtype=torch.float32).to(torch.int64)

    # Extract predicted probabilities as a tensor
    y_probs = torch.tensor(df[class_cols].values, dtype=torch.float32)

    metric_collection = MetricCollection(
        metrics=[
            F1Score(number_of_classes=num_classes, average="macro"),
            F1Score(number_of_classes=num_classes, average="micro"),
            # Precision_K(number_of_classes=num_classes, k=8),
            # Precision_K(number_of_classes=num_classes, k=15),
            # ExactMatchRatio(number_of_classes=num_classes),
            # MeanAveragePrecision(number_of_classes=num_classes),
            # AUC2(number_of_classes=num_classes, average="micro"),
            # AUC2(number_of_classes=num_classes, average="macro"),
        ],
        threshold=model_threshold,
    )

    metric_collection.update(y_probs, targets)
    results_dict = metric_collection.compute(y_probs, targets)

    return results_dict

def permutation_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    a_threshold: float,
    b_threshold: float,
    num_permutations: int = 1000,
    seed: int = 3
) -> dict:
    """
    Perform permutation testing to assess the significance of metric differences
    between two models.

    Args:
        df_a (pd.DataFrame): Predictions from Model A.
        df_b (pd.DataFrame): Predictions from Model B.
        a_threshold (float): Threshold for Model A.
        b_threshold (float): Threshold for Model B.
        num_permutations (int): Number of permutation iterations.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains actual metrics, permutation differences, and p-values.
    """
    
    # Ensure both DataFrames have the same number of samples
    if len(df_a) != len(df_b):
        raise ValueError("DataFrames must have the same number of samples for permutation testing.")

    # Ensure ordering consistency using '_id' if available
    if "_id" in df_a.columns and "_id" in df_b.columns:
        df_a = df_a.sort_values("_id").reset_index(drop=True)
        df_b = df_b.sort_values("_id").reset_index(drop=True)

    excluded_cols = {'_id', 'targets'}
    class_cols = [col for col in df_a.columns if col not in excluded_cols and col in df_b.columns]

    if set(class_cols) != set(df_b.columns) - excluded_cols:
        raise ValueError("Class columns in both DataFrames must be identical.")

    num_samples = len(df_a)

    # Compute actual metrics for both models
    actual_metrics_a = evaluate_predictions(df_a, a_threshold)
    actual_metrics_b = evaluate_predictions(df_b, b_threshold)

    actual_diff = {
        metric: actual_metrics_a[metric] - actual_metrics_b[metric]
        for metric in actual_metrics_a
    }

    perm_diffs = defaultdict(list)

    np.random.seed(seed)

    # Perform permutation iterations
    for _ in tqdm(range(num_permutations), desc="Permutation Testing"):
        swap_mask = np.random.rand(num_samples) < 0.5

        perm_df_a = df_a.copy()
        perm_df_b = df_b.copy()

        # Swap the values in the class columns based on the mask
        perm_df_a.loc[swap_mask, class_cols] = df_b.loc[swap_mask, class_cols].values
        perm_df_b.loc[swap_mask, class_cols] = df_a.loc[swap_mask, class_cols].values

        # Evaluate metrics on the permuted DataFrames
        metrics_a = evaluate_predictions(perm_df_a, a_threshold)
        metrics_b = evaluate_predictions(perm_df_b, b_threshold)

        for metric in actual_diff:
            if metric in metrics_a and metric in metrics_b:
                diff = metrics_a[metric] - metrics_b[metric]
                if isinstance(diff, torch.Tensor):
                    diff = diff.item()
                perm_diffs[metric].append(diff)

    p_values = {}
    for metric, diffs in perm_diffs.items():
        observed_diff = actual_diff.get(metric)
        if observed_diff is None:
            continue
        extreme_count = sum(abs(d) >= abs(observed_diff) for d in diffs)
        p_value = (extreme_count + 1) / (num_permutations + 1)
        p_values[metric] = p_value

    return {
        'actual_metrics_a': actual_metrics_a,
        'actual_metrics_b': actual_metrics_b,
        'actual_diff': actual_diff,
        'perm_diffs': perm_diffs,
        'p_values': p_values
    }


def main():
    """
    Main function to load prediction DataFrames, perform permutation testing,
    and display the results.
    """
    # Paths to the prediction DataFrames
    model_a_path = 'models/entityonly/predictions_test.feather'
    model_b_path = 'models/fulltext/predictions_test.feather'
    
    df_a = pd.read_feather(model_a_path)
    df_b = pd.read_feather(model_b_path)
    
    # Optional: Check for consistency (commented out)
    # if "_id" in df_a.columns and "_id" in df_b.columns:
    #     if not (df_a["_id"].equals(df_b["_id"])):
    #         raise ValueError("The '_id' columns in both DataFrames do not match.")
    # else:
    #     if len(df_a) != len(df_b):
    #         raise ValueError("DataFrames must have the same number of samples.")
    
    # Set the number of permutation iterations and thresholds for each model
    num_permutations = 5  # Adjust the number of iterations as needed
    print(f"Starting permutation testing with {num_permutations} permutations...")
    
    a_threshold = 0.4040403962135315
    b_threshold = 0.4141414165496826
    
    results = permutation_test(
        df_a,
        df_b,
        a_threshold=a_threshold,
        b_threshold=b_threshold,
        num_permutations=num_permutations,
        seed=3
    )


    p_values = results['p_values']
    actual_metrics_a = results['actual_metrics_a']
    actual_metrics_b = results['actual_metrics_b']
    actual_diff = results['actual_diff']
    
    print("\n=== Actual Metrics ===")
    print("Model A Metrics:")
    for metric, value in actual_metrics_a.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nModel B Metrics:")
    for metric, value in actual_metrics_b.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Observed Metric Differences (A - B) ===")
    for metric, diff in actual_diff.items():
        print(f"{metric}: {diff:.4f}")
    
    print("\n=== Permutation Test P-Values ===")
    for metric, p_val in p_values.items():
        significance = "Significant" if p_val < 0.05 else "Not Significant"
        print(f"{metric}: p-value = {p_val:.4f} ({significance})")

if __name__ == "__main__":
    main()
