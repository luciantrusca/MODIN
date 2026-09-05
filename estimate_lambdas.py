import numpy as np
from scipy.special import comb, gammaln
from scipy.stats import norm
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from tqdm import tqdm
import warnings

from piglasso import QJSweeper

warnings.simplefilter('error', OptimizeWarning)

def estimate_lambda_np(edge_counts_all, Q, lambda_range):
    p, _, J = edge_counts_all.shape  # Get the dimensions from edge_counts_all

    # Get the indices for the lower triangular part of the matrix, excluding the diagonal
    lower_tri_indices = np.tril_indices(p, -1)

    # Extract the lower triangular part for each lambda
    N_k_matrix = np.zeros((p * (p - 1) // 2, J))
    for k in range(J):
        N_k_matrix[:, k] = edge_counts_all[:, :, k][lower_tri_indices]

    # Calculate the empirical probability p_k for each edge for each lambda
    p_k_matrix = N_k_matrix / Q
    p_k_matrix = np.clip(p_k_matrix, 1e-5, 1 - 1e-5)  # Regularize probabilities to avoid 0 or 1

    # Calculate theta_matrix using the lower triangular indices for each lambda
    theta_matrix = np.zeros_like(N_k_matrix)
    for k in range(J):
        edge_counts_lambda = N_k_matrix[:, k]
        log_theta = log_comb(Q, edge_counts_lambda) \
                    + edge_counts_lambda * np.log(p_k_matrix[:, k]) \
                    + (Q - edge_counts_lambda) * np.log(1 - p_k_matrix[:, k])
        theta_matrix[:, k] = np.exp(log_theta)

    # Calculate f_k and g for each edge across all lambda values
    f_k_lj_matrix = N_k_matrix / Q
    g_matrix = 4 * f_k_lj_matrix * (1 - f_k_lj_matrix)

    # Reshape the matrices for vectorized operations
    theta_matrix_reshaped = theta_matrix.reshape(-1, J)
    g_matrix_reshaped = g_matrix.reshape(-1, J)

    # Compute the score for each lambda
    scores = np.sum(theta_matrix_reshaped * (1 - g_matrix_reshaped), axis=0)

    # Find the lambda that maximizes the score
    lambda_np = lambda_range[np.argmax(scores)]

    return lambda_np, theta_matrix

def log_comb(n, k):
    """Compute the logarithm of combinations using gamma logarithm for numerical stability."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

# Define a linear function for curve fitting
def linear_func(x, a, b):
    return a * x + b

def fit_lines_and_get_error(index, lambdas, edge_counts, left_bound, right_bound):
    # Only consider data points within the specified bounds
    left_data = lambdas[left_bound:index+1]
    right_data = lambdas[index:right_bound]

    if len(left_data) < 10 or len(right_data) < 10:
        return np.inf

    # Fit lines to the left and right of current index within bounds
    try:
        params_left, _ = curve_fit(linear_func, left_data, edge_counts[left_bound:index+1])
    except:
        print(f'LEFT DATA: problematic curve fit for lambda kneepoints: at lambda index {index}')
        print(f'left indices len: {len(left_data)}')
        params_left = (0,0)
    try:
        params_right, _ = curve_fit(linear_func, right_data, edge_counts[index:right_bound])
    except:
        print(f'RIGHT DATA: problematic curve fit for lambda kneepoints: at lambda index {index}')
        print(f'right indices len: {len(right_data)}')
        params_right = (0,0)

    # Calculate fit errors within bounds
    error_left = np.sum((linear_func(left_data, *params_left) - edge_counts[left_bound:index+1]) ** 2)
    error_right = np.sum((linear_func(right_data, *params_right) - edge_counts[index:right_bound]) ** 2)

    return error_left + error_right

def find_knee_point(lambda_range, edge_counts_all, left_bound, right_bound):
    errors = [fit_lines_and_get_error(i, lambda_range, edge_counts_all, left_bound, right_bound)
              for i in range(left_bound, right_bound)]
    knee_point_index = np.argmin(errors) + left_bound
    return knee_point_index

def find_all_knee_points(lambda_range, edge_counts_all):
    # Sum the edge counts across all nodes
    edge_counts_all = np.sum(edge_counts_all, axis=(0, 1))

    # Find the main knee point across the full range
    main_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, len(lambda_range))
    main_knee_point = lambda_range[main_knee_point_index]

    # For the left knee point, consider points to the left of the main knee point
    left_knee_point_index = find_knee_point(lambda_range, edge_counts_all, 0, main_knee_point_index)
    left_knee_point = lambda_range[left_knee_point_index]

    # For the right knee point, consider points to the right of the main knee point
    # Update the bounds to ensure the fit_lines_and_get_error function considers only the right subset
    right_knee_point_index = find_knee_point(lambda_range, edge_counts_all, main_knee_point_index, len(lambda_range))
    right_knee_point = lambda_range[right_knee_point_index]

    return left_knee_point, main_knee_point, right_knee_point, left_knee_point_index, main_knee_point_index, right_knee_point_index

