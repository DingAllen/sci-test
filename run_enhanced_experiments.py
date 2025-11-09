"""
Enhanced experiment script with sharpness analysis and SAM optimizer.
Extends the original experiments to investigate the generalization gap.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from test_functions import get_test_function
from optimizers import SGD, SGDMomentum, AdaGrad, RMSprop, Adam, AdamW, SAM, get_optimizer
from experiment_runner import ExperimentRunner
from visualization import Visualizer
from sharpness_analysis import (
    compute_sharpness_metrics,
    analyze_sharpness_convergence_correlation
)
import matplotlib.pyplot as plt
import seaborn as sns


def run_sharpness_experiments():
    """Run experiments with sharpness analysis."""
    print("="*80)
    print("ENHANCED EXPERIMENTS: SHARPNESS ANALYSIS AND SAM OPTIMIZER")
    print("="*80)
    
    # Initialize
    runner = ExperimentRunner(results_dir='results')
    viz = Visualizer(figures_dir='figures')
    
    # Test function to focus on (Rastrigin - most challenging)
    func_name = 'rastrigin'
    func = get_test_function(func_name, dim=2)
    
    # Define optimizers including SAM variants
    optimizers = {
        'SGD': (SGD, {'learning_rate': 0.01}),
        'SGD-Momentum': (SGDMomentum, {'learning_rate': 0.01, 'momentum': 0.9}),
        'Adam': (Adam, {'learning_rate': 0.01}),
        'SAM-SGD': (lambda **kwargs: SAM(SGD(**{k:v for k,v in kwargs.items() if k != 'rho'}), rho=kwargs.get('rho', 0.05)), 
                    {'learning_rate': 0.01, 'rho': 0.05}),
        'SAM-Adam': (lambda **kwargs: SAM(Adam(**{k:v for k,v in kwargs.items() if k != 'rho'}), rho=kwargs.get('rho', 0.05)), 
                     {'learning_rate': 0.01, 'rho': 0.05}),
    }
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: Sharpness Analysis on {func_name.upper()}")
    print(f"{'='*80}")
    
    results_dict = {}
    sharpness_dict = {}
    
    for opt_name, (opt_class, opt_params) in optimizers.items():
        print(f"\nRunning {opt_name} on {func_name}...")
        
        # Run multiple experiments
        results = runner.run_multiple_runs(
            opt_class, opt_params, func,
            n_runs=20, max_iterations=1000, random_seed=42
        )
        results_dict[opt_name] = results
        
        # Compute sharpness for each run
        print(f"  Computing sharpness metrics...")
        sharpness_values = []
        final_losses = []
        max_eigenvalues = []
        neighborhood_sharpness = []
        
        for i, result in enumerate(results[:10]):  # First 10 runs for speed
            x_final = result['final_x']
            metrics = compute_sharpness_metrics(func, x_final, verbose=False)
            
            sharpness_values.append(metrics['neighborhood_sharpness'])
            max_eigenvalues.append(metrics['max_eigenvalue'])
            final_losses.append(result['final_loss'])
            neighborhood_sharpness.append(metrics['neighborhood_sharpness'])
        
        sharpness_dict[opt_name] = {
            'mean_max_eigenvalue': np.mean(max_eigenvalues),
            'std_max_eigenvalue': np.std(max_eigenvalues),
            'mean_neighborhood_sharpness': np.mean(neighborhood_sharpness),
            'std_neighborhood_sharpness': np.std(neighborhood_sharpness),
            'mean_final_loss': np.mean(final_losses),
            'std_final_loss': np.std(final_losses),
            'max_eigenvalues': max_eigenvalues,
            'neighborhood_sharpness': neighborhood_sharpness,
            'final_losses': final_losses,
        }
        
        print(f"  Mean final loss: {np.mean(final_losses):.4f} ± {np.std(final_losses):.4f}")
        print(f"  Mean max eigenvalue: {np.mean(max_eigenvalues):.4f} ± {np.std(max_eigenvalues):.4f}")
        print(f"  Mean neighborhood sharpness: {np.mean(neighborhood_sharpness):.4f}")
        
        # Save results
        runner.save_results(results, f'enhanced_{func_name}_{opt_name.lower().replace("-", "_")}_results.pkl')
    
    # Generate sharpness comparison visualizations
    print(f"\nGenerating sharpness visualizations...")
    
    # 1. Sharpness comparison bar plot
    create_sharpness_comparison_plot(sharpness_dict, func_name)
    
    # 2. Sharpness vs. Loss scatter plot
    create_sharpness_loss_scatter(sharpness_dict, func_name)
    
    # 3. Generate convergence curves including SAM
    viz.plot_convergence_curves(
        results_dict,
        filename=f'enhanced_convergence_{func_name}_with_sam.png'
    )
    
    # 4. Distribution comparison
    viz.plot_final_loss_distribution(
        results_dict,
        filename=f'enhanced_distribution_{func_name}_with_sam.png'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SHARPNESS ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nSharpness Rankings (Lower is Flatter):")
    sorted_by_sharpness = sorted(sharpness_dict.items(), 
                                  key=lambda x: x[1]['mean_max_eigenvalue'])
    for rank, (opt_name, stats) in enumerate(sorted_by_sharpness, 1):
        print(f"{rank}. {opt_name}: Max Eigenvalue = {stats['mean_max_eigenvalue']:.4f}, "
              f"Final Loss = {stats['mean_final_loss']:.4f}")
    
    print("\n" + "="*80)
    print("ENHANCED EXPERIMENTS COMPLETED!")
    print("="*80)
    
    return results_dict, sharpness_dict


def create_sharpness_comparison_plot(sharpness_dict, func_name):
    """Create bar plot comparing sharpness across optimizers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    opt_names = list(sharpness_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(opt_names)))
    
    # Plot 1: Max Eigenvalue
    max_eigs = [sharpness_dict[name]['mean_max_eigenvalue'] for name in opt_names]
    max_eig_stds = [sharpness_dict[name]['std_max_eigenvalue'] for name in opt_names]
    
    axes[0].bar(opt_names, max_eigs, yerr=max_eig_stds, color=colors, alpha=0.7, capsize=5)
    axes[0].set_ylabel('Max Hessian Eigenvalue', fontsize=11)
    axes[0].set_title('Loss Landscape Sharpness (Hessian-based)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Neighborhood Sharpness
    nb_sharp = [sharpness_dict[name]['mean_neighborhood_sharpness'] for name in opt_names]
    nb_sharp_stds = [sharpness_dict[name]['std_neighborhood_sharpness'] for name in opt_names]
    
    axes[1].bar(opt_names, nb_sharp, yerr=nb_sharp_stds, color=colors, alpha=0.7, capsize=5)
    axes[1].set_ylabel('Neighborhood Sharpness', fontsize=11)
    axes[1].set_title('Loss Landscape Sharpness (Neighborhood-based)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Sharpness Comparison on {func_name.capitalize()} Function', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join('figures', f'enhanced_sharpness_comparison_{func_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


def create_sharpness_loss_scatter(sharpness_dict, func_name):
    """Create scatter plot showing sharpness vs. final loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'SGD': '#1f77b4', 'SGD-Momentum': '#ff7f0e', 'Adam': '#9467bd',
              'SAM-SGD': '#2ca02c', 'SAM-Adam': '#d62728'}
    
    # Plot 1: Max Eigenvalue vs. Loss
    for opt_name, data in sharpness_dict.items():
        color = colors.get(opt_name, '#7f7f7f')
        axes[0].scatter(data['max_eigenvalues'], data['final_losses'],
                       label=opt_name, alpha=0.6, s=60, color=color)
    
    axes[0].set_xlabel('Max Hessian Eigenvalue (Sharpness)', fontsize=11)
    axes[0].set_ylabel('Final Loss', fontsize=11)
    axes[0].set_title('Sharpness vs. Final Loss (Hessian)', fontsize=12)
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Neighborhood Sharpness vs. Loss
    for opt_name, data in sharpness_dict.items():
        color = colors.get(opt_name, '#7f7f7f')
        axes[1].scatter(data['neighborhood_sharpness'], data['final_losses'],
                       label=opt_name, alpha=0.6, s=60, color=color)
    
    axes[1].set_xlabel('Neighborhood Sharpness', fontsize=11)
    axes[1].set_ylabel('Final Loss', fontsize=11)
    axes[1].set_title('Sharpness vs. Final Loss (Neighborhood)', fontsize=12)
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle(f'Sharpness-Performance Relationship on {func_name.capitalize()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join('figures', f'enhanced_sharpness_vs_loss_{func_name}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


if __name__ == '__main__':
    results_dict, sharpness_dict = run_sharpness_experiments()
