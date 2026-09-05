import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (GLASSO)
project_dir = os.path.dirname(script_dir)
# Add the project directory to the Python path
sys.path.append(project_dir)
# Change the working directory to the project directory
os.chdir(project_dir)

import numpy as np
from random import sample
import random
from scipy.special import comb
from sklearn.covariance import empirical_covariance
import os
import rpy2.situation
os.environ["R_HOME"] = rpy2.situation.get_r_home()

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from tqdm import tqdm
import warnings

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

class QJSweeper:
    """
    Class for parallel optimisation of the piGGM objective function, across Q sub-samples and J lambdas.

    Attributes
    ----------
    data : array-like, shape (n, p)
        The data matrix.
    prior_matrix : array-like, shape (p, p)
        The prior matrix. Used to identify which edges are penalized by lambda_wp.
    p : int
        The number of variables.

    Methods
    -------
    optimize_for_q_and_j(single_subsamp_idx, lambdax)
        Fits weighted graphical LASSO for a given sub-sample and lambda value.

    run_subsample_optimization(lambda_range)
        Runs optimize_for_q_and_j across all sub-samples and the full lambda range, accumulating edge counts.
    """
    def __init__(self, data, prior_matrix, b, Q, rank=1, size=1, seed=42):
        self.data = data
        self.prior_matrix = prior_matrix
        self.p = data.shape[1]
        self.n = data.shape[0]
        self.Q = Q
        self.subsample_indices = self.get_subsamples_indices(self.n, b, Q, rank, size, seed=42)

    def get_subsamples_indices(self, n, b, Q, rank, size, seed=42):
        """
        Generate a unique set of subsamples indices for a given MPI rank and size.
        """
        # Error handling: check if b and Q are valid 
        if b >= n:
            raise ValueError("b should be less than the number of samples n.")
        if Q > comb(n, b, exact=True):
            raise ValueError("Q should be smaller or equal to the number of possible sub-samples.")

        random.seed(seed + rank)  # Ensure each rank gets different subsamples
        subsamples_indices = set()

        # Each rank will attempt to generate Q/size unique subsamples
        subsamples_per_rank = Q // size
        attempts = 0
        max_attempts = 10e+5  # to avoid an infinite loop

        while len(subsamples_indices) < subsamples_per_rank and attempts < max_attempts:
            # Generate a random combination
            new_comb = tuple(sorted(sample(range(n), b)))
            subsamples_indices.add(new_comb)
            attempts += 1

        if attempts == max_attempts:
            raise Exception(f"Rank {rank}: Max attempts reached when generating subsamples.")

        return list(subsamples_indices)

    def optimize_for_q_and_j(self, single_subsamp_idx, lambdax):
        """
        Optimizes the objective function for a given sub-sample (q) and lambda (j).
        Parameters
        ----------
        single_subsamp_idx : array-like, shape (b)
            The indices of the sub-sample.
        lambdax : float
            The lambda value.

        Returns
        -------
        edge_counts : array-like, shape (p, p)
            Binary matrix of which edges are present in the fitted precision matrix.
        precision_matrix : array-like, shape (p, p)
            The fitted precision matrix (all zeros if the R glasso call failed).
        success : int
            1 if the fit succeeded, 0 otherwise.
        """
        data = self.data
        p = self.p
        prior_matrix = self.prior_matrix
        sub_sample = data[np.array(single_subsamp_idx), :]
        S = empirical_covariance(sub_sample)

        # Number of observations
        nobs = sub_sample.shape[0]

        # Penalty matrix (adapt this to your actual penalty matrix logic)
        penalty_matrix = lambdax * np.ones((p,p))

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
                edge_counts = (np.abs(precision_matrix) > 1e-5).astype(int)
                return edge_counts, precision_matrix, 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr, flush=True)
            return np.zeros((p, p)), np.zeros((p, p)), 0



    def run_subsample_optimization(self, lambda_range):
        """
        Run optimization on the subsamples for the entire lambda range.
        """
        edge_counts_all = np.zeros((self.p, self.p, len(lambda_range)))
        success_counts = np.zeros(len(lambda_range))

        for q_idx in tqdm(self.subsample_indices):
            for lambdax in lambda_range:
                edge_counts, precision_matrix, success_check = self.optimize_for_q_and_j(q_idx, lambdax)
                l_idx = np.where(lambda_range == lambdax)[0][0]
                edge_counts_all[:, :, l_idx] += edge_counts
                success_counts[l_idx] += success_check

        return edge_counts_all, success_counts

