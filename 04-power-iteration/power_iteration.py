import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    r_0 = np.ones(data.shape[0], dtype=float)

    # result[0] - eigenvalue
    # result[1] - eigenvector
    result = [1., r_0]

    for k in range(num_steps):
        temp = np.dot(data, result[1])
        result[0] = np.sum(result[1] * temp) / np.sum(result[1] ** 2)
        result[1] = temp / np.sqrt(np.sum(temp ** 2))

    return float(result[0]), result[1]