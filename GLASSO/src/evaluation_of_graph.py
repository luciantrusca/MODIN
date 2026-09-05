import sys
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from sklearn.covariance import empirical_covariance
from src.glasso_installation import check_and_install_glasso



# Activate the automatic conversion of numpy objects to R objects
numpy2ri.activate()

# Check and install glasso if needed before defining the R function
if not check_and_install_glasso():
    raise RuntimeError("Failed to ensure glasso package is available")

# Define the R function for weighted graphical lasso
ro.r('''
weighted_glasso <- function(data, penalty_matrix, nobs) {
    # Suppress warnings and messages when loading the library
    suppressWarnings(suppressMessages(library(glasso, quietly = TRUE)))
    
    tryCatch({
        result <- glasso(s = as.matrix(data), rho = penalty_matrix, nobs = nobs)
        return(list(precision_matrix = result$wi, edge_counts = result$wi != 0))
    }, error = function(e) {
        return(list(error_message = toString(e$message)))
    })
}
''')


def optimize_graph(data, prior_matrix, lambda_np, lambda_wp, verbose=False):
    """
    Optimizes the objective function using the entire data set and the estimated lambda.

    Parameters
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix.
    lambda_val : float
        The regularization parameter for the edges.

    Returns
    -------
    opt_precision_mat : array-like, shape (p, p)
        The optimized precision matrix.
    """
    # Number of observations
    nobs = data.shape[0]
    p = data.shape[1]

    complete_graph_edges = (p * (p - 1)) / 2

    try:
        S = empirical_covariance(data)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return np.zeros((p, p)), np.zeros((p, p)), 0


    # generate penalty matrix, where values = lambda_np for non-prior edges and lambda_wp for prior edges
    penalty_matrix = np.zeros_like(prior_matrix, dtype=np.float64)

    # Assign penalties based on the prior matrix
    penalty_matrix[prior_matrix != 0] = lambda_wp
    penalty_matrix[prior_matrix == 0] = lambda_np

    # # fill diagonal with 0s
    np.fill_diagonal(penalty_matrix, 0)

    # check for NaNs or Infs in penalty matrix
    if np.isnan(penalty_matrix).any():
        print('NaNs in penalty matrix')
    if np.isinf(penalty_matrix).any():
        print('Infs in penalty matrix')


    # Call the R function from Python
    weighted_glasso = ro.globalenv['weighted_glasso']
    try:
        result = weighted_glasso(S, penalty_matrix, nobs)   
        # Check for an error message returned from R
        if 'error_message' in result.names:
            error_message = result.rx('error_message')[0][0]
            print(f"R Error: {error_message}", file=sys.stderr, flush=True)
            return np.zeros((p, p)), np.zeros((p, p)), 0
        else:
            precision_matrix = np.array(result.rx('precision_matrix')[0])
            # rho_ij = -theta_ij / sqrt(theta_ii * theta_jj) needs the actual diagonal,
            # so normalize before zeroing it out below.
            _d = np.sqrt(np.diag(precision_matrix))
            precision_matrix = -precision_matrix / np.outer(_d, _d)
            np.fill_diagonal(precision_matrix, 0)
            edge_counts = np.sum((np.abs(precision_matrix) > 1e-5).astype(int)) / 2
            density = edge_counts / complete_graph_edges
            return precision_matrix, edge_counts, density
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
        return np.zeros((p, p)), np.zeros((p, p)), 0
