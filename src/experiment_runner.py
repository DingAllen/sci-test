"""
Experiment runner for optimization algorithm comparison.
Handles running experiments, logging results, and saving data.
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle


class ExperimentRunner:
    """Run optimization experiments and collect results."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def run_single_experiment(self, optimizer, objective_fn, x0, 
                             max_iterations=1000, tolerance=1e-6,
                             record_trajectory=True):
        """
        Run a single optimization experiment.
        
        Args:
            optimizer: Optimizer instance
            objective_fn: Objective function to minimize
            x0: Initial point
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            record_trajectory: Whether to record full trajectory
            
        Returns:
            Dictionary with experiment results
        """
        optimizer.reset()
        x = x0.copy()
        
        trajectory = [x.copy()] if record_trajectory else []
        loss_history = [objective_fn(x)]
        
        for iteration in range(max_iterations):
            # Compute gradient
            gradient = objective_fn.gradient(x)
            
            # Take optimization step
            x = optimizer.step(x, gradient)
            
            # Record
            loss = objective_fn(x)
            loss_history.append(loss)
            if record_trajectory:
                trajectory.append(x.copy())
            
            # Check convergence
            if len(loss_history) > 1 and abs(loss_history[-1] - loss_history[-2]) < tolerance:
                break
        
        return {
            'final_x': x,
            'final_loss': loss_history[-1],
            'loss_history': loss_history,
            'trajectory': trajectory if record_trajectory else None,
            'iterations': len(loss_history) - 1,
            'converged': abs(loss_history[-1] - objective_fn.optimal_value) < 0.1
        }
    
    def run_multiple_runs(self, optimizer_class, optimizer_params, 
                         objective_fn, n_runs=30, max_iterations=1000,
                         random_seed=42):
        """
        Run multiple experiments with different random initializations.
        
        Args:
            optimizer_class: Optimizer class
            optimizer_params: Parameters for optimizer
            objective_fn: Objective function
            n_runs: Number of runs with different initializations
            max_iterations: Maximum iterations per run
            random_seed: Random seed for reproducibility
            
        Returns:
            List of results dictionaries
        """
        np.random.seed(random_seed)
        results = []
        
        bounds = objective_fn.bounds
        dim = objective_fn.dim
        
        for run in range(n_runs):
            # Random initialization within bounds
            x0 = np.random.uniform(bounds[0], bounds[1], size=dim)
            
            # Create fresh optimizer instance
            optimizer = optimizer_class(**optimizer_params)
            
            # Run experiment
            result = self.run_single_experiment(
                optimizer, objective_fn, x0, 
                max_iterations=max_iterations,
                record_trajectory=(run < 5)  # Only record trajectory for first 5 runs
            )
            result['run'] = run
            result['x0'] = x0
            results.append(result)
        
        return results
    
    def run_learning_rate_sweep(self, optimizer_class, objective_fn,
                                learning_rates, n_runs_per_lr=10,
                                max_iterations=1000, random_seed=42):
        """
        Run experiments across different learning rates.
        
        Args:
            optimizer_class: Optimizer class
            objective_fn: Objective function
            learning_rates: List of learning rates to test
            n_runs_per_lr: Number of runs per learning rate
            max_iterations: Maximum iterations per run
            random_seed: Random seed
            
        Returns:
            Dictionary mapping learning rates to results
        """
        all_results = {}
        
        for lr in tqdm(learning_rates, desc=f"LR sweep for {optimizer_class.__name__}"):
            optimizer_params = {'learning_rate': lr}
            results = self.run_multiple_runs(
                optimizer_class, optimizer_params, objective_fn,
                n_runs=n_runs_per_lr, max_iterations=max_iterations,
                random_seed=random_seed
            )
            all_results[lr] = results
        
        return all_results
    
    def save_results(self, results, filename):
        """Save results to file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved to {filepath}")
    
    def load_results(self, filename):
        """Load results from file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_summary_statistics(self, results, filename):
        """Save summary statistics as JSON."""
        stats = {
            'mean_final_loss': float(np.mean([r['final_loss'] for r in results])),
            'std_final_loss': float(np.std([r['final_loss'] for r in results])),
            'median_final_loss': float(np.median([r['final_loss'] for r in results])),
            'min_final_loss': float(np.min([r['final_loss'] for r in results])),
            'max_final_loss': float(np.max([r['final_loss'] for r in results])),
            'mean_iterations': float(np.mean([r['iterations'] for r in results])),
            'success_rate': float(np.mean([r['converged'] for r in results])),
            'n_runs': len(results)
        }
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
