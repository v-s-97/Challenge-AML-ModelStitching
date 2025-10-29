import numpy as np  


def mrr(pred_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    """
    Compute Mean Reciprocal Rank (MRR)
    Args:
        pred_indices: (N, K) array of predicted indices for N queries (top-K)
        gt_indices: (N,) array of ground truth indices
    Returns:
        mrr: Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i] == gt_indices[i])[0]
        if matches.size > 0:
            reciprocal_ranks.append(1.0 / (matches[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def recall_at_k(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int) -> float:
    """Compute Recall@k
    Args:
        pred_indices: (N, N) array of top indices for N queries
        gt_indices: (N,) array of ground truth indices
        k: number of top predictions to consider
    Returns:
        recall: Recall@k
    """
    recall = 0
    for i in range(len(gt_indices)):
        if gt_indices[i] in pred_indices[i, :k]:
            recall += 1
    recall /= len(gt_indices)
    return recall

import numpy as np

def ndcg(pred_indices: np.ndarray, gt_indices: np.ndarray, k: int = 100) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k)
    Args:
        pred_indices: (N, K) array of predicted indices for N queries
        gt_indices: (N,) array of ground truth indices
        k: number of top predictions to consider
    Returns:
        ndcg: NDCG@k
    """
    ndcg_total = 0.0
    for i in range(len(gt_indices)):
        matches = np.where(pred_indices[i, :k] == gt_indices[i])[0]
        if matches.size > 0:
            rank = matches[0] + 1
            ndcg_total += 1.0 / np.log2(rank + 1)  # DCG (IDCG = 1)
    return ndcg_total / len(gt_indices)


