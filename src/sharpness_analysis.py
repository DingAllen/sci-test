"""
Sharpness analysis module for measuring loss landscape geometry.
"""

import numpy as np
from scipy.linalg import eigh


def compute_hessian_eigenvalues(objective_fn, x, n_top=5):
    """
    Compute top eigenvalues of the Hessian matrix using finite differences.
    
    Args:
        objective_fn: Objective function
        x: Point at which to compute Hessian
        n_top: Number of top eigenvalues to return
        
    Returns:
        Array of top eigenvalues (sorted descending)
    """
    epsilon = 1e-5
    n = len(x)
    hessian = np.zeros((n, n))
    
    # Compute Hessian using central differences
    f_center = objective_fn(x)
    
    for i in range(n):
        for j in range(i, n):
            # f(x + e_i + e_j)
            x_pp = x.copy()
            x_pp[i] += epsilon
            x_pp[j] += epsilon
            f_pp = objective_fn(x_pp)
            
            # f(x + e_i - e_j)
            x_pm = x.copy()
            x_pm[i] += epsilon
            x_pm[j] -= epsilon
            f_pm = objective_fn(x_pm)
            
            # f(x - e_i + e_j)
            x_mp = x.copy()
            x_mp[i] -= epsilon
            x_mp[j] += epsilon
            f_mp = objective_fn(x_mp)
            
            # f(x - e_i - e_j)
            x_mm = x.copy()
            x_mm[i] -= epsilon
            x_mm[j] -= epsilon
            f_mm = objective_fn(x_mm)
            
            # Second derivative
            h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon * epsilon)
            hessian[i, j] = h_ij
            if i != j:
                hessian[j, i] = h_ij
    
    # Compute eigenvalues
    eigenvalues = eigh(hessian, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
    
    return eigenvalues[:n_top]


def compute_neighborhood_sharpness(objective_fn, x, n_samples=50, radius=0.01):
    """
    Compute sharpness by measuring loss increase in a neighborhood.
    
    Sharpness = max(L(x + δ) - L(x)) / ||δ||²
    
    Args:
        objective_fn: Objective function
        x: Point at which to measure sharpness
        n_samples: Number of random perturbations to sample
        radius: Maximum perturbation radius
        
    Returns:
        Sharpness value (higher = sharper minimum)
    """
    f_center = objective_fn(x)
    max_sharpness = 0.0
    
    for _ in range(n_samples):
        # Random perturbation
        delta = np.random.randn(len(x))
        delta = delta / np.linalg.norm(delta) * radius * np.random.rand()
        
        # Evaluate perturbed loss
        f_perturbed = objective_fn(x + delta)
        
        # Compute normalized loss increase
        delta_norm_sq = np.dot(delta, delta)
        if delta_norm_sq > 1e-10:
            sharpness = (f_perturbed - f_center) / delta_norm_sq
            max_sharpness = max(max_sharpness, sharpness)
    
    return max_sharpness


def compute_average_neighborhood_loss(objective_fn, x, n_samples=100, radius=0.01):
    """
    Compute average loss in a small neighborhood around x.
    
    Args:
        objective_fn: Objective function
        x: Center point
        n_samples: Number of samples
        radius: Neighborhood radius
        
    Returns:
        Average loss in neighborhood
    """
    f_center = objective_fn(x)
    losses = [f_center]
    
    for _ in range(n_samples):
        # Random perturbation
        delta = np.random.randn(len(x))
        delta = delta / np.linalg.norm(delta) * radius * np.random.rand()
        
        # Evaluate perturbed loss
        f_perturbed = objective_fn(x + delta)
        losses.append(f_perturbed)
    
    return np.mean(losses), np.std(losses)


def compute_sharpness_metrics(objective_fn, x, verbose=False):
    """
    Compute multiple sharpness metrics for a converged solution.
    
    Args:
        objective_fn: Objective function
        x: Converged solution
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with various sharpness metrics
    """
    metrics = {}
    
    # 1. Hessian-based metrics
    if verbose:
        print("  Computing Hessian eigenvalues...")
    eigenvalues = compute_hessian_eigenvalues(objective_fn, x, n_top=min(5, len(x)))
    metrics['max_eigenvalue'] = eigenvalues[0]
    metrics['top_5_eigenvalues'] = eigenvalues
    metrics['eigenvalue_sum'] = np.sum(np.abs(eigenvalues))
    
    # 2. Neighborhood-based sharpness
    if verbose:
        print("  Computing neighborhood sharpness...")
    metrics['neighborhood_sharpness'] = compute_neighborhood_sharpness(
        objective_fn, x, n_samples=50, radius=0.01
    )
    
    # 3. Average neighborhood loss
    if verbose:
        print("  Computing average neighborhood loss...")
    avg_loss, std_loss = compute_average_neighborhood_loss(
        objective_fn, x, n_samples=100, radius=0.01
    )
    metrics['avg_neighborhood_loss'] = avg_loss
    metrics['std_neighborhood_loss'] = std_loss
    
    # 4. Center loss for reference
    metrics['center_loss'] = objective_fn(x)
    
    return metrics


def analyze_sharpness_convergence_correlation(results_dict, objective_fn):
    """
    Analyze correlation between sharpness and final loss across runs.
    
    Args:
        results_dict: Dictionary mapping optimizer names to results
        objective_fn: Objective function
        
    Returns:
        Dictionary with correlation statistics
    """
    analysis = {}
    
    for opt_name, results in results_dict.items():
        sharpness_values = []
        final_losses = []
        
        for result in results:
            x_final = result['final_x']
            final_loss = result['final_loss']
            
            # Compute sharpness
            metrics = compute_sharpness_metrics(objective_fn, x_final, verbose=False)
            sharpness = metrics['max_eigenvalue']
            
            sharpness_values.append(sharpness)
            final_losses.append(final_loss)
        
        # Compute correlation
        correlation = np.corrcoef(sharpness_values, final_losses)[0, 1]
        
        analysis[opt_name] = {
            'mean_sharpness': np.mean(sharpness_values),
            'std_sharpness': np.std(sharpness_values),
            'sharpness_loss_correlation': correlation,
            'sharpness_values': sharpness_values,
            'final_losses': final_losses
        }
    
    return analysis
