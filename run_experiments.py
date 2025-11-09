"""
Main experiment script - runs all experiments and generates visualizations.
This script implements the complete experimental protocol.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from test_functions import get_test_function
from optimizers import SGD, SGDMomentum, AdaGrad, RMSprop, Adam, AdamW
from experiment_runner import ExperimentRunner
from visualization import Visualizer


def main():
    """Run all experiments."""
    print("="*80)
    print("COMPARATIVE ANALYSIS OF DEEP LEARNING OPTIMIZATION ALGORITHMS")
    print("="*80)
    
    # Initialize
    runner = ExperimentRunner(results_dir='results')
    viz = Visualizer(figures_dir='figures')
    
    # Define test functions
    test_functions = {
        'rastrigin': get_test_function('rastrigin', dim=2),
        'ackley': get_test_function('ackley', dim=2),
        'rosenbrock': get_test_function('rosenbrock', dim=2),
        'beale': get_test_function('beale'),
    }
    
    # Define optimizers with default hyperparameters
    optimizers = {
        'SGD': (SGD, {'learning_rate': 0.01}),
        'SGD-Momentum': (SGDMomentum, {'learning_rate': 0.01, 'momentum': 0.9}),
        'AdaGrad': (AdaGrad, {'learning_rate': 0.1}),
        'RMSprop': (RMSprop, {'learning_rate': 0.01}),
        'Adam': (Adam, {'learning_rate': 0.01}),
        'AdamW': (AdamW, {'learning_rate': 0.01}),
    }
    
    # Experiment 1: Generate 3D surface plots for all test functions
    print("\n" + "="*80)
    print("EXPERIMENT 1: Generating 3D Surface Visualizations")
    print("="*80)
    
    for func_name, func in test_functions.items():
        print(f"\nGenerating 3D surface for {func_name}...")
        viz.plot_3d_surface(func, filename=f'surface_3d_{func_name}.png', resolution=100)
    
    # Experiment 2: Run optimization with trajectories for visualization
    print("\n" + "="*80)
    print("EXPERIMENT 2: Optimization Trajectory Visualization")
    print("="*80)
    
    for func_name, func in test_functions.items():
        print(f"\nRunning trajectory experiments for {func_name}...")
        trajectories = {}
        
        for opt_name, (opt_class, opt_params) in optimizers.items():
            # Use same starting point for fair comparison
            np.random.seed(42)
            x0 = np.random.uniform(func.bounds[0], func.bounds[1], size=func.dim)
            
            optimizer = opt_class(**opt_params)
            result = runner.run_single_experiment(
                optimizer, func, x0, 
                max_iterations=500,
                record_trajectory=True
            )
            trajectories[opt_name] = result['trajectory']
            print(f"  {opt_name}: Final loss = {result['final_loss']:.6f}, "
                  f"Iterations = {result['iterations']}")
        
        # Plot trajectories
        viz.plot_2d_landscape_with_trajectories(
            func, trajectories, 
            filename=f'trajectories_{func_name}.png'
        )
    
    # Experiment 3: Multiple runs for statistical analysis
    print("\n" + "="*80)
    print("EXPERIMENT 3: Statistical Analysis with Multiple Runs")
    print("="*80)
    
    all_summary_stats = {}
    
    for func_name, func in test_functions.items():
        print(f"\n{'='*60}")
        print(f"Function: {func_name.upper()}")
        print(f"{'='*60}")
        
        results_dict = {}
        summary_stats_dict = {}
        
        for opt_name, (opt_class, opt_params) in optimizers.items():
            print(f"\nRunning {opt_name} on {func_name}...")
            results = runner.run_multiple_runs(
                opt_class, opt_params, func,
                n_runs=30, max_iterations=1000, random_seed=42
            )
            results_dict[opt_name] = results
            
            # Save results
            runner.save_results(results, f'{func_name}_{opt_name.lower()}_results.pkl')
            
            # Compute and save summary statistics
            stats = runner.save_summary_statistics(
                results, f'{func_name}_{opt_name.lower()}_stats.json'
            )
            summary_stats_dict[opt_name] = stats
            
            print(f"  Mean final loss: {stats['mean_final_loss']:.6f} ± {stats['std_final_loss']:.6f}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"  Mean iterations: {stats['mean_iterations']:.1f}")
        
        # Store for later use
        all_summary_stats[func_name] = summary_stats_dict
        
        # Generate visualizations for this function
        print(f"\nGenerating visualizations for {func_name}...")
        viz.plot_convergence_curves(
            results_dict, 
            filename=f'convergence_{func_name}.png'
        )
        viz.plot_final_loss_distribution(
            results_dict,
            filename=f'distribution_{func_name}.png'
        )
        viz.plot_comparison_bar_chart(
            summary_stats_dict,
            filename=f'comparison_{func_name}.png'
        )
    
    # Experiment 4: Learning rate sensitivity analysis
    print("\n" + "="*80)
    print("EXPERIMENT 4: Learning Rate Sensitivity Analysis")
    print("="*80)
    
    # Focus on Rastrigin function for LR sweep
    func = test_functions['rastrigin']
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    
    lr_results_dict = {}
    
    for opt_name, (opt_class, opt_params) in optimizers.items():
        print(f"\nRunning LR sweep for {opt_name}...")
        # Remove learning_rate from opt_params as it will be swept
        params_no_lr = {k: v for k, v in opt_params.items() if k != 'learning_rate'}
        
        lr_sweep_results = runner.run_learning_rate_sweep(
            opt_class, func, learning_rates,
            n_runs_per_lr=10, max_iterations=1000, random_seed=42
        )
        lr_results_dict[opt_name] = lr_sweep_results
        
        # Save LR sweep results
        runner.save_results(
            lr_sweep_results, 
            f'lr_sweep_{opt_name.lower()}_rastrigin.pkl'
        )
    
    # Visualize LR sensitivity
    print("\nGenerating LR sensitivity visualizations...")
    viz.plot_learning_rate_heatmap(
        lr_results_dict, learning_rates,
        filename='lr_sensitivity_mean_loss.png',
        metric='mean_loss'
    )
    viz.plot_learning_rate_heatmap(
        lr_results_dict, learning_rates,
        filename='lr_sensitivity_success_rate.png',
        metric='success_rate'
    )
    
    # Generate summary report
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nOverall Performance Ranking (based on Rastrigin function):")
    rastrigin_stats = all_summary_stats['rastrigin']
    sorted_opts = sorted(rastrigin_stats.items(), 
                        key=lambda x: x[1]['mean_final_loss'])
    
    for rank, (opt_name, stats) in enumerate(sorted_opts, 1):
        print(f"{rank}. {opt_name}: Mean Loss = {stats['mean_final_loss']:.6f}, "
              f"Success Rate = {stats['success_rate']*100:.1f}%")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nResults saved in: {runner.results_dir}/")
    print(f"Figures saved in: {viz.figures_dir}/")
    print("\nReady for paper writing phase.")


if __name__ == '__main__':
    main()
