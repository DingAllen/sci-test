"""
Visualization module for generating publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Color palette for optimizers
OPTIMIZER_COLORS = {
    'SGD': '#1f77b4',
    'SGD-Momentum': '#ff7f0e',
    'AdaGrad': '#2ca02c',
    'RMSprop': '#d62728',
    'Adam': '#9467bd',
    'AdamW': '#8c564b'
}


class Visualizer:
    """Generate visualizations for optimization experiments."""
    
    def __init__(self, figures_dir='figures'):
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
    
    def plot_2d_landscape_with_trajectories(self, objective_fn, trajectories_dict,
                                           filename='landscape_trajectories.png',
                                           resolution=100):
        """
        Plot 2D contour of objective function with optimization trajectories.
        
        Args:
            objective_fn: Objective function
            trajectories_dict: Dict mapping optimizer names to trajectory arrays
            filename: Output filename
            resolution: Grid resolution for contour plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create mesh grid
        bounds = objective_fn.bounds
        x1 = np.linspace(bounds[0], bounds[1], resolution)
        x2 = np.linspace(bounds[0], bounds[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Compute function values
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = objective_fn(np.array([X1[i, j], X2[i, j]]))
        
        # Plot contours
        levels = np.logspace(np.log10(np.min(Z) + 0.1), np.log10(np.max(Z)), 20)
        contour = ax.contour(X1, X2, Z, levels=levels, cmap='gray', alpha=0.3, linewidths=0.5)
        contourf = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis', alpha=0.3)
        
        # Plot global optimum
        ax.plot(objective_fn.global_optimum[0], objective_fn.global_optimum[1], 
                'r*', markersize=15, label='Global Optimum', zorder=5)
        
        # Plot trajectories
        for opt_name, trajectory in trajectories_dict.items():
            trajectory = np.array(trajectory)
            color = OPTIMIZER_COLORS.get(opt_name, None)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   '-o', label=opt_name, alpha=0.7, markersize=3,
                   color=color, linewidth=1.5)
            # Mark starting point
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'x', 
                   markersize=8, color=color, markeredgewidth=2)
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(f'{objective_fn.__class__.__name__} - Optimization Trajectories')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.2)
        
        plt.colorbar(contourf, ax=ax, label='Function Value')
        plt.tight_layout()
        
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_convergence_curves(self, results_dict, filename='convergence_curves.png',
                               log_scale=True):
        """
        Plot convergence curves for different optimizers.
        
        Args:
            results_dict: Dict mapping optimizer names to list of results
            filename: Output filename
            log_scale: Whether to use log scale for y-axis
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for opt_name, results in results_dict.items():
            # Get mean and std of loss histories
            max_len = max(len(r['loss_history']) for r in results)
            loss_matrix = np.full((len(results), max_len), np.nan)
            
            for i, result in enumerate(results):
                history = result['loss_history']
                loss_matrix[i, :len(history)] = history
            
            # Compute statistics
            mean_loss = np.nanmean(loss_matrix, axis=0)
            std_loss = np.nanstd(loss_matrix, axis=0)
            
            iterations = np.arange(len(mean_loss))
            color = OPTIMIZER_COLORS.get(opt_name, None)
            
            ax.plot(iterations, mean_loss, label=opt_name, color=color, linewidth=2)
            ax.fill_between(iterations, mean_loss - std_loss, mean_loss + std_loss,
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss Value')
        ax.set_title('Convergence Curves (Mean ± Std)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_final_loss_distribution(self, results_dict, 
                                     filename='final_loss_distribution.png'):
        """
        Plot distribution of final loss values as box plots.
        
        Args:
            results_dict: Dict mapping optimizer names to list of results
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        colors = []
        
        for opt_name, results in results_dict.items():
            final_losses = [r['final_loss'] for r in results]
            data.append(final_losses)
            labels.append(opt_name)
            colors.append(OPTIMIZER_COLORS.get(opt_name, 'gray'))
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                       notch=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Final Loss Value')
        ax.set_title('Distribution of Final Loss Values')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_3d_surface(self, objective_fn, filename='3d_surface.png',
                       resolution=100, elev=30, azim=45):
        """
        Plot 3D surface of the objective function.
        
        Args:
            objective_fn: Objective function (must be 2D)
            filename: Output filename
            resolution: Grid resolution
            elev: Elevation angle
            azim: Azimuth angle
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        bounds = objective_fn.bounds
        x1 = np.linspace(bounds[0], bounds[1], resolution)
        x2 = np.linspace(bounds[0], bounds[1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        Z = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = objective_fn(np.array([X1[i, j], X2[i, j]]))
        
        # Clip extreme values for better visualization
        Z_clipped = np.clip(Z, np.percentile(Z, 1), np.percentile(Z, 99))
        
        surf = ax.plot_surface(X1, X2, Z_clipped, cmap='viridis', 
                              alpha=0.9, edgecolor='none')
        
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('f(x)')
        ax.set_title(f'{objective_fn.__class__.__name__} - 3D Surface')
        ax.view_init(elev=elev, azim=azim)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_learning_rate_heatmap(self, lr_results_dict, learning_rates,
                                   filename='lr_heatmap.png', metric='mean_loss'):
        """
        Plot heatmap of performance across learning rates.
        
        Args:
            lr_results_dict: Dict mapping optimizer names to LR sweep results
            learning_rates: List of learning rates tested
            filename: Output filename
            metric: Metric to visualize ('mean_loss' or 'success_rate')
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        optimizer_names = list(lr_results_dict.keys())
        matrix = np.zeros((len(optimizer_names), len(learning_rates)))
        
        for i, opt_name in enumerate(optimizer_names):
            lr_sweep = lr_results_dict[opt_name]
            for j, lr in enumerate(learning_rates):
                if lr in lr_sweep:
                    results = lr_sweep[lr]
                    if metric == 'mean_loss':
                        matrix[i, j] = np.mean([r['final_loss'] for r in results])
                    elif metric == 'success_rate':
                        matrix[i, j] = np.mean([r['converged'] for r in results])
        
        # Use log scale for mean_loss
        if metric == 'mean_loss':
            matrix_plot = np.log10(matrix + 1e-10)
            cmap = 'viridis_r'
            cbar_label = 'log10(Mean Final Loss)'
        else:
            matrix_plot = matrix
            cmap = 'RdYlGn'
            cbar_label = 'Success Rate'
        
        im = ax.imshow(matrix_plot, aspect='auto', cmap=cmap)
        
        ax.set_xticks(np.arange(len(learning_rates)))
        ax.set_yticks(np.arange(len(optimizer_names)))
        ax.set_xticklabels([f'{lr:.4f}' for lr in learning_rates], rotation=45, ha='right')
        ax.set_yticklabels(optimizer_names)
        
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Optimizer')
        ax.set_title(f'Performance Across Learning Rates - {metric.replace("_", " ").title()}')
        
        plt.colorbar(im, ax=ax, label=cbar_label)
        plt.tight_layout()
        
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_comparison_bar_chart(self, summary_stats_dict, 
                                  filename='comparison_bar_chart.png'):
        """
        Plot bar chart comparing optimizer performance metrics.
        
        Args:
            summary_stats_dict: Dict mapping optimizer names to summary statistics
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        optimizer_names = list(summary_stats_dict.keys())
        colors = [OPTIMIZER_COLORS.get(name, 'gray') for name in optimizer_names]
        
        # Mean final loss
        mean_losses = [summary_stats_dict[name]['mean_final_loss'] 
                      for name in optimizer_names]
        axes[0, 0].bar(optimizer_names, mean_losses, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Mean Final Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Mean Final Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Success rate
        success_rates = [summary_stats_dict[name]['success_rate'] * 100
                        for name in optimizer_names]
        axes[0, 1].bar(optimizer_names, success_rates, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Convergence Success Rate')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim(0, 100)
        
        # Mean iterations
        mean_iters = [summary_stats_dict[name]['mean_iterations']
                     for name in optimizer_names]
        axes[1, 0].bar(optimizer_names, mean_iters, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Mean Iterations')
        axes[1, 0].set_title('Mean Iterations to Convergence')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Std final loss
        std_losses = [summary_stats_dict[name]['std_final_loss']
                     for name in optimizer_names]
        axes[1, 1].bar(optimizer_names, std_losses, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Std of Final Loss')
        axes[1, 1].set_title('Robustness (Lower is Better)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
