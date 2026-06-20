import numpy as np
from typing import Tuple, List


def _get_procrustes_parameters(x: np.ndarray, target: np.ndarray, translation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Procrustes transformation to align x to target.

    Parameters:
        x (np.ndarray): The input matrix to be aligned.
        target (np.ndarray): The target matrix to align to.
        translation (bool, optional): Whether to compute a translation component (default False).

    Returns:
        output (Tuple[np.ndarray, np.ndarray]): A tuple containing the rotation matrix and the translation vector.
    """
    if translation:
        m_target = np.mean(target, axis=0)
        m_x = np.mean(x, axis=0)
        c_target = target - m_target
        M = c_target.T @ x
        U, _, Vt = np.linalg.svd(M)
        rotation_matrix = Vt.T @ U.T
        translation_vector = m_target - rotation_matrix.T @ m_x
    else:
        M = target.T @ x
        U, _, Vt = np.linalg.svd(M)
        rotation_matrix = Vt.T @ U.T
        translation_vector = np.zeros(x.shape[1])
    return rotation_matrix, translation_vector


def perform_procrustes(x: np.ndarray, target: np.ndarray, matrix_to_transform: np.ndarray, translation: bool = False) -> np.ndarray:
    """
    Align matrix_to_transform to target using a Procrustes transformation.

    Parameters:
        x (np.ndarray): The reference matrix for alignment.
        target (np.ndarray): The target matrix for alignment.
        matrix_to_transform (np.ndarray): The matrix to be transformed.
        translation (bool, optional): Whether to include translation in the alignment (default False).

    Returns:
        transformed_matrix (np.ndarray): The transformed matrix after alignment.

    Raises:
        ValueError: If x and target do not have the same shape or if x and 
                matrix_to_transform do not have the same number of columns.
    """
    if x.shape != target.shape:
        raise ValueError("x and target do not have the same shape")
    if x.shape[1] != matrix_to_transform.shape[1]:
        raise ValueError(
            "x and matrix_to_transform do not have the same number of columns")

    rotation_matrix, translation_vector = _get_procrustes_parameters(
        x, target, translation)
    return matrix_to_transform @ rotation_matrix + translation_vector


def get_partitions_for_divide_conquer(n: int, l: int, c_points: int, r: int) -> List[np.ndarray]:
    """
    Divide indices 0,...,n-1 into partitions suitable for the divide and conquer algorithm for dimensionality reduction techniques.

    Parameters:
        n (int): Total number of data points.
        l (int): Partition size.
        c_points (int): Number of common points.
        r (int): Target dimensionality (used for validation).

    Returns:
        idx_list (List[np.ndarray]): A list of partitions, each represented as a numpy array of indices.

    Raises:
        ValueError: If l is not greater than c_points, if l - c_points is less than c_points, or if l - c_points is less than r+2.
    """
    # Input validation
    if l <= c_points:
        raise ValueError("l must be greater than c_points")
    if l - c_points < c_points:
        raise ValueError("l-c_points must be at least c_points")
    if l - c_points < r+2:
        raise ValueError("l-c_points must be at least r+2")

    # Generate random permutation
    permutation = np.random.permutation(n)

    # Handle case when data fits in a single partition
    if n <= l:
        return [permutation]

    # Divide into multiple partitions
    p = np.ceil((n - l) / (l - c_points))
    first_partition = permutation[:l]
    other_partitions = np.array_split(permutation[l:], p)

    return [first_partition] + other_partitions
