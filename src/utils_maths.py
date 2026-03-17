import numpy as np
from numpy.linalg import inv

# Utility functions for computing the bounds
def rho_fn(P: np.ndarray) -> float:
    """
        rho(P) = max eigenvalue of P
    """
    return max(abs(np.linalg.eigvals(P)))

def w_norm(x: np.ndarray, P: np.ndarray) -> float:
    """
        sqrt( x^T P x)
    """
    return np.sqrt(np.inner(P @ x, x))

def logdet_part(P: np.ndarray, V: np.ndarray):
    """
        log|I + P^{-1} V|
    """
    Pinv = inv(P)
    eigenvalues = np.linalg.eigvals(Pinv @ V)
    eigenvalues = np.real(eigenvalues)  # in case of numerical issues, we take the real part
    if np.any(eigenvalues <= -0.01):
        return np.inf
    else:
        return np.log(1. + eigenvalues).sum()
    
def op_norm(A: np.ndarray, W = None) -> float:
    """
        spectral norm of A:
         sqrt(rho(A W A^T)) = max singular value of A
    """
    if W is None:
        return np.sqrt(rho_fn(A @ A.T))
    else:
        return np.sqrt(rho_fn(A @ W @ A.T))
