"""
Occurrence-Based Double Reordering (ODR) for Energy-Efficient CAM Mapping

This module implements the ODR algorithm which optimizes memory utilization
for decision tree to CAM conversion by:
1. Sorting conditions (features) by occurrence frequency (descending)
2. Reordering paths so those with rare conditions appear first
3. Concentrating X-state (don't-care) cells in bottom-right TCAMs for removal

This approach maintains energy efficiency by keeping each path in a fixed row,
avoiding computational overhead during result retrieval.
"""

import numpy as np
from sklearn import tree
from dt2cam import DT2Array, DTLoader


def applyODR(
    thresholdArray: np.ndarray,
    col2featureID: np.ndarray,
    row2classID: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Occurrence-Based Double Reordering to threshold array.

    Algorithm 1: ODR
    1. Count occurrence frequency of each condition (feature column)
    2. Sort conditions by frequency (descending)
    3. Reorder paths: place paths with rare conditions first

    Args:
        thresholdArray: shape (num_paths, num_features, 2) [lowerBound, upperBound]
        col2featureID: mapping from column index to feature ID
        row2classID: mapping from row index to class ID

    Returns:
        Reordered thresholdArray, col2featureID, row2classID
    """
    # Step 1: Count condition occurrences (non-NaN cells per column)
    condition_occurrences = _countConditionOccurrences(thresholdArray)

    # Step 2: Sort conditions by occurrence frequency (descending)
    sorted_col_indices = np.argsort(condition_occurrences)[::-1]

    # Step 3: Reorder paths based on rare conditions
    sorted_row_indices = _reorderPathsByRarity(thresholdArray, sorted_col_indices)

    # Apply reordering
    thresholdArray_reordered = thresholdArray[sorted_row_indices, :, :]
    thresholdArray_reordered = thresholdArray_reordered[:, sorted_col_indices, :]
    col2featureID_reordered = col2featureID[sorted_col_indices]
    row2classID_reordered = row2classID[sorted_row_indices]

    return thresholdArray_reordered, col2featureID_reordered, row2classID_reordered


def _countConditionOccurrences(thresholdArray: np.ndarray) -> np.ndarray:
    """
    Count how many paths use each condition (feature column).

    A condition is "used" in a path if the cell is not all NaN.

    Args:
        thresholdArray: shape (num_paths, num_features, 2)

    Returns:
        Array of shape (num_features,) with occurrence counts
    """
    num_features = thresholdArray.shape[1]
    occurrences = np.zeros(num_features, dtype=int)

    for col_idx in range(num_features):
        # Count rows where this column has at least one non-NaN value
        for row_idx in range(thresholdArray.shape[0]):
            if not (np.isnan(thresholdArray[row_idx, col_idx, 0]) and
                    np.isnan(thresholdArray[row_idx, col_idx, 1])):
                occurrences[col_idx] += 1

    return occurrences


def _reorderPathsByRarity(
    thresholdArray: np.ndarray,
    sorted_col_indices: np.ndarray
) -> np.ndarray:
    """
    Reorder paths so those containing rare conditions appear first.

    Following Algorithm 1:
    - Iterate through conditions from rarest to most common
    - For each condition, collect paths that contain it (in order)
    - Remove collected paths from pool

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        sorted_col_indices: column indices sorted by frequency (descending)

    Returns:
        Array of row indices representing the new path order
    """
    num_paths = thresholdArray.shape[0]
    pool = set(range(num_paths))  # Remaining path indices
    ordered_paths = []

    # Iterate from rarest to most common condition
    for col_idx in reversed(sorted_col_indices):
        paths_with_condition = []

        # Find paths that contain this condition
        for row_idx in pool:
            if not (np.isnan(thresholdArray[row_idx, col_idx, 0]) and
                    np.isnan(thresholdArray[row_idx, col_idx, 1])):
                paths_with_condition.append(row_idx)

        # Add these paths to ordered list and remove from pool
        ordered_paths.extend(paths_with_condition)
        pool -= set(paths_with_condition)

    return np.array(ordered_paths, dtype=int)


def DT2Array_ODR(
    DT: tree.DecisionTreeClassifier | DTLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Convert Decision Tree to CAM array with ODR optimization.

    This is a wrapper around DT2Array that applies ODR reordering
    to optimize memory utilization.

    Args:
        DT: sklearn DecisionTreeClassifier or DTLoader

    Returns:
        thresholdArray: ODR-optimized threshold array
        col2featureID: reordered feature ID mapping
        row2classID: reordered class ID mapping
        thresholdMin: minimum threshold value
        thresholdMax: maximum threshold value
    """
    # Get initial arrays from standard conversion
    thresholdArray, col2featureID, row2classID, thresholdMin, thresholdMax = DT2Array(DT)

    # Apply ODR reordering
    thresholdArray, col2featureID, row2classID = applyODR(
        thresholdArray, col2featureID, row2classID
    )

    return thresholdArray, col2featureID, row2classID, thresholdMin, thresholdMax


def calculateSparsity(thresholdArray: np.ndarray) -> float:
    """
    Calculate the sparsity (fraction of X-state/NaN cells) in the threshold array.

    Args:
        thresholdArray: shape (num_paths, num_features, 2)

    Returns:
        Sparsity as a fraction between 0 and 1
    """
    total_cells = thresholdArray.shape[0] * thresholdArray.shape[1]
    empty_cells = 0

    for row_idx in range(thresholdArray.shape[0]):
        for col_idx in range(thresholdArray.shape[1]):
            if (np.isnan(thresholdArray[row_idx, col_idx, 0]) and
                np.isnan(thresholdArray[row_idx, col_idx, 1])):
                empty_cells += 1

    return empty_cells / total_cells


def calculateEnergyMetrics(thresholdArray: np.ndarray, num_tcams: int = 1) -> dict:
    """
    Calculate energy consumption metrics for CAM arrays.

    Energy model:
    - Each active TCAM cell consumes energy during lookup
    - X-state (don't-care) cells in bottom-right can be power-gated (removed)
    - Multiple TCAMs consume energy independently

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        num_tcams: number of TCAM chips used

    Returns:
        Dictionary with energy metrics
    """
    num_paths, num_features, _ = thresholdArray.shape

    # Count active (non-NaN) cells
    active_cells = 0
    for row_idx in range(num_paths):
        for col_idx in range(num_features):
            if not (np.isnan(thresholdArray[row_idx, col_idx, 0]) and
                    np.isnan(thresholdArray[row_idx, col_idx, 1])):
                active_cells += 1

    total_cells = num_paths * num_features

    # Energy model (relative units)
    # Assume each active TCAM cell consumes 1 unit of energy per lookup
    energy_per_cell_per_lookup = 1.0

    # Base energy: all cells active
    base_energy = total_cells * energy_per_cell_per_lookup

    # Actual energy: only active cells consume power
    actual_energy = active_cells * energy_per_cell_per_lookup

    # TCAM overhead: multiple TCAMs have additional routing/control energy
    tcam_overhead_per_chip = total_cells * 0.05  # 5% overhead per TCAM
    total_tcam_overhead = tcam_overhead_per_chip * num_tcams

    # Total energy with overhead
    total_energy = actual_energy + total_tcam_overhead

    return {
        'total_cells': total_cells,
        'active_cells': active_cells,
        'inactive_cells': total_cells - active_cells,
        'num_tcams': num_tcams,
        'base_energy': base_energy,
        'active_cell_energy': actual_energy,
        'tcam_overhead_energy': total_tcam_overhead,
        'total_energy': total_energy,
        'energy_savings': base_energy - total_energy,
        'energy_efficiency': (1 - total_energy / base_energy) * 100,  # percentage
    }


# ============================================================================
# Similarity-Based Path Clustering (SPC)
# ============================================================================


def _getPathConditions(thresholdArray: np.ndarray, path_idx: int) -> set:
    """
    Get the set of conditions (feature indices) used by a specific path.

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        path_idx: index of the path

    Returns:
        Set of feature indices that have non-NaN values for this path
    """
    conditions = set()
    for col_idx in range(thresholdArray.shape[1]):
        if not (np.isnan(thresholdArray[path_idx, col_idx, 0]) and
                np.isnan(thresholdArray[path_idx, col_idx, 1])):
            conditions.add(col_idx)
    return conditions


def _getClusterConditions(thresholdArray: np.ndarray, cluster_paths: list) -> set:
    """
    Get all unique conditions used by paths in a cluster.

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        cluster_paths: list of path indices in the cluster

    Returns:
        Set of all feature indices used by any path in the cluster
    """
    if not cluster_paths:
        return set()

    all_conditions = set()
    for path_idx in cluster_paths:
        all_conditions.update(_getPathConditions(thresholdArray, path_idx))
    return all_conditions


def _calcSimilarity(
    thresholdArray: np.ndarray,
    pool_paths: list,
    cluster_paths: list
) -> dict:
    """
    Calculate similarity metrics for each path in pool with current cluster.

    Similarity is based on:
    1. Number of shared conditions (higher is better)
    2. Number of unique conditions after adding path (lower is better)

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        pool_paths: list of path indices still in pool
        cluster_paths: list of path indices in current cluster

    Returns:
        Dictionary mapping path_idx -> (shared_count, unique_count_after)
    """
    cluster_conditions = _getClusterConditions(thresholdArray, cluster_paths)
    similarity = {}

    for path_idx in pool_paths:
        path_conditions = _getPathConditions(thresholdArray, path_idx)

        # Count shared conditions
        shared_count = len(path_conditions & cluster_conditions)

        # Count unique conditions after adding this path
        unique_count_after = len(cluster_conditions | path_conditions)

        similarity[path_idx] = (shared_count, unique_count_after)

    return similarity


def _findBestCandidate(
    similarity: dict,
    S: int,
    cluster_paths: list
) -> int | None:
    """
    Find the best candidate path to add to current cluster.

    Selection criteria (in order of priority):
    1. Resulting unique conditions must not exceed S
    2. Maximize shared conditions
    3. Minimize resulting unique conditions

    Args:
        similarity: dict mapping path_idx -> (shared_count, unique_count_after)
        S: TCAM size constraint
        cluster_paths: current cluster paths

    Returns:
        Best candidate path index, or None if no suitable candidate exists
    """
    if not similarity:
        return None

    # Filter candidates that would exceed capacity
    valid_candidates = {
        path_idx: (shared, unique)
        for path_idx, (shared, unique) in similarity.items()
        if unique <= S
    }

    if not valid_candidates:
        return None

    # Sort by: 1) maximize shared conditions, 2) minimize unique conditions
    best_candidate = max(
        valid_candidates.items(),
        key=lambda x: (x[1][0], -x[1][1])  # (shared_count, -unique_count_after)
    )

    return best_candidate[0]


def applySPC(
    thresholdArray: np.ndarray,
    col2featureID: np.ndarray,
    row2classID: np.ndarray,
    S: int
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """
    Apply Similarity-Based Path Clustering to organize paths into TCAMs.

    Algorithm 2: SPC
    Greedily clusters paths to minimize TCAM count while maximizing similarity
    within each cluster. Each cluster becomes a separate TCAM.

    Args:
        thresholdArray: shape (num_paths, num_features, 2)
        col2featureID: mapping from column index to feature ID
        row2classID: mapping from row index to class ID
        S: TCAM size (max paths and max unique conditions per cluster)

    Returns:
        - List of clusters, each containing (thresholdArray, col2featureID, row2classID)
        - Total number of TCAMs required
    """
    num_paths = thresholdArray.shape[0]
    pool = list(range(num_paths))  # Path indices still to be assigned
    clusters = []
    current_cluster = []
    current_num = 0

    while pool:
        # Calculate similarity between remaining paths and current cluster
        similarity = _calcSimilarity(thresholdArray, pool, current_cluster)

        # Find best candidate to add to current cluster
        candidate = _findBestCandidate(similarity, S, current_cluster)

        # If no suitable candidate or cluster is full, finalize current cluster
        if candidate is None or current_num == S:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = []
            current_num = 0
        else:
            # Add candidate to current cluster
            current_cluster.append(candidate)
            current_num += 1
            pool.remove(candidate)

    # Add final cluster if not empty
    if current_cluster:
        clusters.append(current_cluster)

    # Convert clusters to separate TCAM arrays
    tcam_clusters = []
    for cluster_paths in clusters:
        # Extract paths in this cluster
        cluster_threshold = thresholdArray[cluster_paths, :, :]
        cluster_classID = row2classID[cluster_paths]

        # Extract only the features used by this cluster
        cluster_conditions = _getClusterConditions(thresholdArray, cluster_paths)
        used_features = sorted(list(cluster_conditions))

        if used_features:
            cluster_threshold = cluster_threshold[:, used_features, :]
            cluster_featureID = col2featureID[used_features]
        else:
            cluster_featureID = col2featureID

        tcam_clusters.append((cluster_threshold, cluster_featureID, cluster_classID))

    return tcam_clusters, len(clusters)


def DT2Array_SPC(
    DT: tree.DecisionTreeClassifier | DTLoader,
    S: int = 32
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int, float, float]:
    """
    Convert Decision Tree to CAM arrays with SPC optimization.

    This approach clusters similar paths together to minimize the number of
    TCAMs required, improving both memory and energy efficiency.

    Args:
        DT: sklearn DecisionTreeClassifier or DTLoader
        S: TCAM size constraint (default: 32)

    Returns:
        - List of TCAM clusters, each containing (thresholdArray, col2featureID, row2classID)
        - Number of TCAMs required
        - thresholdMin: minimum threshold value
        - thresholdMax: maximum threshold value
    """
    # Get initial arrays from standard conversion
    thresholdArray, col2featureID, row2classID, thresholdMin, thresholdMax = DT2Array(DT)

    # Apply SPC clustering
    tcam_clusters, num_tcams = applySPC(
        thresholdArray, col2featureID, row2classID, S
    )

    return tcam_clusters, num_tcams, thresholdMin, thresholdMax


def run_example(dataset_name: str, X, y, max_depth: int = 5):
    """Run ODR and SPC examples on a given dataset."""

    # Train a decision tree
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, y)

    # Standard conversion
    print("\n" + "=" * 70)
    print(f"Dataset: {dataset_name}")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("Standard DT2Array Conversion")
    print("=" * 70)
    thresholdArray, col2featureID, row2classID, tMin, tMax = DT2Array(clf)
    sparsity_original = calculateSparsity(thresholdArray)
    num_paths = thresholdArray.shape[0]
    num_features = thresholdArray.shape[1]
    print(f"Number of paths (leaf nodes): {num_paths}")
    print(f"Number of features used: {num_features}")
    print(f"Shape: {thresholdArray.shape}")
    print(f"Sparsity (original): {sparsity_original:.3f}")
    print(f"Feature order: {col2featureID}")
    print(f"Class order: {row2classID}")

    # Energy metrics for standard conversion
    energy_original = calculateEnergyMetrics(thresholdArray, num_tcams=1)
    print(f"\nEnergy Metrics:")
    print(f"  Active cells: {energy_original['active_cells']} / {energy_original['total_cells']}")
    print(f"  Total energy: {energy_original['total_energy']:.2f} units")
    print(f"  Energy efficiency: {energy_original['energy_efficiency']:.2f}%")

    # ODR conversion
    print("\n" + "=" * 70)
    print("ODR-Optimized Conversion (Occurrence-Based Double Reordering)")
    print("=" * 70)
    thresholdArray_odr, col2featureID_odr, row2classID_odr, tMin, tMax = DT2Array_ODR(clf)
    sparsity_odr = calculateSparsity(thresholdArray_odr)
    print(f"Shape: {thresholdArray_odr.shape}")
    print(f"Sparsity (ODR): {sparsity_odr:.3f}")
    print(f"Feature order (by frequency): {col2featureID_odr}")
    print(f"Class order (rare paths first): {row2classID_odr}")
    print(f"Note: ODR concentrates X-state cells in bottom-right corner")

    # Energy metrics for ODR
    energy_odr = calculateEnergyMetrics(thresholdArray_odr, num_tcams=1)
    print(f"\nEnergy Metrics (ODR):")
    print(f"  Active cells: {energy_odr['active_cells']} / {energy_odr['total_cells']}")
    print(f"  Total energy: {energy_odr['total_energy']:.2f} units")
    print(f"  Energy efficiency: {energy_odr['energy_efficiency']:.2f}%")
    print(f"  Energy savings vs original: {((energy_original['total_energy'] - energy_odr['total_energy']) / energy_original['total_energy'] * 100):.2f}%")

    # SPC conversion
    print("\n" + "=" * 70)
    print("SPC-Optimized Conversion (Similarity-Based Path Clustering)")
    print("=" * 70)
    S = 32  # TCAM size constraint
    tcam_clusters, num_tcams, tMin, tMax = DT2Array_SPC(clf, S=S)

    print(f"TCAM size constraint (S): {S}")
    print(f"Number of TCAMs required: {num_tcams}")
    print(f"Naive mapping would require: 1 TCAM (all paths in one)")
    print(f"\nCluster breakdown:")

    total_cells = 0
    total_empty_cells = 0

    for i, (cluster_threshold, cluster_featureID, cluster_classID) in enumerate(tcam_clusters):
        cluster_sparsity = calculateSparsity(cluster_threshold)
        cluster_cells = cluster_threshold.shape[0] * cluster_threshold.shape[1]
        cluster_empty = int(cluster_sparsity * cluster_cells)

        total_cells += cluster_cells
        total_empty_cells += cluster_empty

        print(f"  Cluster {i+1}:")
        print(f"    Shape: {cluster_threshold.shape}")
        print(f"    Paths: {len(cluster_classID)}")
        print(f"    Features used: {len(cluster_featureID)}")
        print(f"    Sparsity: {cluster_sparsity:.3f}")
        print(f"    Classes: {cluster_classID}")

    overall_sparsity_spc = total_empty_cells / total_cells if total_cells > 0 else 0
    print(f"\nOverall SPC sparsity: {overall_sparsity_spc:.3f}")
    print(f"Memory efficiency: Paths distributed across {num_tcams} optimized TCAMs")
    print(f"Key benefit: Minimizes wasted capacity by clustering similar paths")

    # Calculate total energy for SPC (sum across all clusters)
    total_spc_energy = 0
    total_spc_active_cells = 0
    for cluster_threshold, _, _ in tcam_clusters:
        cluster_energy = calculateEnergyMetrics(cluster_threshold, num_tcams=1)
        total_spc_energy += cluster_energy['total_energy']
        total_spc_active_cells += cluster_energy['active_cells']

    print(f"\nEnergy Metrics (SPC):")
    print(f"  Total active cells across all TCAMs: {total_spc_active_cells}")
    print(f"  Total energy: {total_spc_energy:.2f} units")
    print(f"  Energy savings vs original: {((energy_original['total_energy'] - total_spc_energy) / energy_original['total_energy'] * 100):.2f}%")
    print(f"  Energy savings vs ODR: {((energy_odr['total_energy'] - total_spc_energy) / energy_odr['total_energy'] * 100):.2f}%")

    # Comparison summary
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    original_capacity = num_paths * num_features
    spc_capacity = total_cells

    print(f"Original approach:")
    print(f"  - Single TCAM: {num_paths} × {num_features} = {original_capacity} cells")
    print(f"  - Sparsity: {sparsity_original:.3f}")
    print(f"  - Energy: {energy_original['total_energy']:.2f} units (baseline)")
    print(f"\nODR approach:")
    print(f"  - Single reordered TCAM: {num_paths} × {num_features} = {original_capacity} cells")
    print(f"  - Sparsity: {sparsity_odr:.3f}")
    print(f"  - Energy: {energy_odr['total_energy']:.2f} units ({((energy_original['total_energy'] - energy_odr['total_energy']) / energy_original['total_energy'] * 100):.2f}% savings)")
    print(f"  - Benefit: Enables removal of bottom-right TCAMs, same energy as original (reordering only)")
    print(f"\nSPC approach:")
    print(f"  - {num_tcams} clustered TCAMs: total {spc_capacity} cells")
    print(f"  - Sparsity: {overall_sparsity_spc:.3f}")
    print(f"  - Memory reduction: {(1 - spc_capacity/original_capacity)*100:.1f}%")
    print(f"  - Energy: {total_spc_energy:.2f} units ({((energy_original['total_energy'] - total_spc_energy) / energy_original['total_energy'] * 100):.2f}% savings)")
    print(f"  - Benefit: Maximizes utilization, minimizes TCAM count AND reduces energy")


if __name__ == "__main__":
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import LabelEncoder

    # Example 1: Iris dataset
    print("\n" + "#" * 70)
    print("# EXAMPLE 1: IRIS DATASET")
    print("#" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    run_example("Iris", X_iris, y_iris, max_depth=5)

    # Example 2: Credit Approval dataset
    print("\n\n" + "#" * 70)
    print("# EXAMPLE 2: CREDIT APPROVAL DATASET")
    print("#" * 70)

    try:
        # Load Credit Approval dataset from UCI repository
        # Dataset: https://archive.ics.uci.edu/dataset/27/credit+approval
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

        # Column names (from dataset description)
        column_names = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
            'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'class'
        ]

        df = pd.read_csv(url, names=column_names, na_values='?')

        # Handle missing values - drop rows with missing values for simplicity
        df = df.dropna()

        # Separate features and target
        X_credit = df.drop('class', axis=1)
        y_credit = df['class']

        # Encode categorical variables
        le = LabelEncoder()
        for col in X_credit.columns:
            if X_credit[col].dtype == 'object':
                X_credit[col] = le.fit_transform(X_credit[col])

        # Encode target variable (+ and -)
        y_credit = le.fit_transform(y_credit)

        X_credit = X_credit.values

        print(f"Credit Approval Dataset loaded: {X_credit.shape[0]} samples, {X_credit.shape[1]} features")

        run_example("Credit Approval", X_credit, y_credit, max_depth=5)

    except Exception as e:
        print(f"Could not load Credit Approval dataset: {e}")
        print("Skipping Credit Approval example...")
