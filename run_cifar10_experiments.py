"""
Neural network experiments on CIFAR-10 to study generalization gap.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torch.utils.data import DataLoader


class ResNet18(nn.Module):
    """Simplified ResNet-18 for CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Use pretrained ResNet18 adapted for CIFAR-10
        import torchvision.models as models
        self.model = models.resnet18(pretrained=False, num_classes=num_classes)
        # Modify first conv layer for 32x32 input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    def forward(self, x):
        return self.model(x)


class SAMOptimizer:
    """
    PyTorch SAM optimizer wrapper.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho = rho
        
    def first_step(self, zero_grad=False):
        """Compute adversarial perturbation and move to that point."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w
                
        if zero_grad:
            self.zero_grad()
    
    def second_step(self, zero_grad=False):
        """Step back and update with the gradient at perturbed point."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # update with the gradient at perturbed point
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        """Combined step for compatibility."""
        assert closure is not None, "SAM requires closure"
        
        # First forward-backward pass
        closure()
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass
        closure()
        self.second_step()
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    @property
    def state(self):
        return self.base_optimizer.state


def train_epoch(model, train_loader, criterion, optimizer, device, is_sam=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        if is_sam:
            # SAM requires closure
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            
            optimizer.step(closure)
            
            # Get metrics
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total


def test(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(test_loader), 100. * correct / total


def compute_model_sharpness(model, data_loader, criterion, device, rho=0.01, n_samples=10):
    """
    Compute sharpness of trained model using perturbation-based method.
    """
    model.eval()
    
    # Get baseline loss
    baseline_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            baseline_loss += loss.item()
    baseline_loss /= len(data_loader)
    
    # Compute sharpness via random perturbations
    max_increase = 0.0
    
    for _ in range(n_samples):
        # Random perturbation
        perturbed_state = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                noise = torch.randn_like(param) * rho
                perturbed_state[name] = param.data.clone()
                param.data.add_(noise)
                total_norm += noise.norm().item() ** 2
        
        total_norm = np.sqrt(total_norm)
        
        # Compute perturbed loss
        perturbed_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                perturbed_loss += loss.item()
        perturbed_loss /= len(data_loader)
        
        # Restore original weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = perturbed_state[name]
        
        # Compute normalized sharpness
        if total_norm > 1e-10:
            sharpness = (perturbed_loss - baseline_loss) / (total_norm ** 2)
            max_increase = max(max_increase, sharpness)
    
    return max_increase


def run_cifar10_experiment(optimizer_name, lr, epochs=50, seed=42):
    """
    Run a single CIFAR-10 experiment with specified optimizer.
    
    Note: For full experiments use epochs=200, but using 50 for faster demonstration.
    """
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name} (lr={lr}, seed={seed})")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # For sharpness computation, use a smaller subset
    small_testset = torch.utils.data.Subset(testset, range(1000))
    small_test_loader = DataLoader(small_testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    is_sam = 'SAM' in optimizer_name
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == 'SAM-SGD':
        optimizer = SAMOptimizer(model.parameters(), optim.SGD, rho=0.05, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'SAM-Adam':
        optimizer = SAMOptimizer(model.parameters(), optim.Adam, rho=0.05, lr=lr, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer if is_sam else optimizer, T_max=epochs)
    
    # Training
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'generalization_gap': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, is_sam)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['generalization_gap'].append(train_acc - test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Gap: {train_acc-test_acc:.2f}%')
        
        scheduler.step()
    
    # Compute final sharpness
    print("Computing model sharpness...")
    sharpness = compute_model_sharpness(model, small_test_loader, criterion, device, rho=0.01, n_samples=5)
    
    final_results = {
        'optimizer': optimizer_name,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'final_generalization_gap': history['generalization_gap'][-1],
        'best_test_acc': max(history['test_acc']),
        'sharpness': sharpness,
        'history': history
    }
    
    print(f"\nFinal Results:")
    print(f"  Train Acc: {final_results['final_train_acc']:.2f}%")
    print(f"  Test Acc: {final_results['final_test_acc']:.2f}%")
    print(f"  Generalization Gap: {final_results['final_generalization_gap']:.2f}%")
    print(f"  Best Test Acc: {final_results['best_test_acc']:.2f}%")
    print(f"  Sharpness: {final_results['sharpness']:.6f}")
    
    # Save results
    os.makedirs('results/cifar10', exist_ok=True)
    with open(f'results/cifar10/{optimizer_name.lower()}_seed{seed}.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


def run_all_cifar10_experiments():
    """Run CIFAR-10 experiments for all optimizers."""
    print("="*80)
    print("CIFAR-10 EXPERIMENTS: GENERALIZATION GAP ANALYSIS")
    print("="*80)
    print("\nNote: Running with 50 epochs for demonstration.")
    print("For full results, use 200 epochs in production.\n")
    
    optimizers_configs = [
        ('SGD', 0.1),
        ('Adam', 0.001),
        ('SAM-SGD', 0.1),
        ('SAM-Adam', 0.001),
    ]
    
    all_results = {}
    
    for opt_name, lr in optimizers_configs:
        results = run_cifar10_experiment(opt_name, lr, epochs=50, seed=42)
        all_results[opt_name] = results
    
    # Create summary visualizations
    create_cifar10_visualizations(all_results)
    
    # Print summary
    print("\n" + "="*80)
    print("CIFAR-10 EXPERIMENTS SUMMARY")
    print("="*80)
    
    print("\nTest Accuracy Rankings:")
    sorted_by_acc = sorted(all_results.items(), key=lambda x: x[1]['best_test_acc'], reverse=True)
    for rank, (opt_name, results) in enumerate(sorted_by_acc, 1):
        print(f"{rank}. {opt_name}: Test Acc = {results['best_test_acc']:.2f}%, "
              f"Gap = {results['final_generalization_gap']:.2f}%, "
              f"Sharpness = {results['sharpness']:.6f}")
    
    print("\nSharpness Rankings (Lower is Better):")
    sorted_by_sharp = sorted(all_results.items(), key=lambda x: x[1]['sharpness'])
    for rank, (opt_name, results) in enumerate(sorted_by_sharp, 1):
        print(f"{rank}. {opt_name}: Sharpness = {results['sharpness']:.6f}")
    
    return all_results


def create_cifar10_visualizations(all_results):
    """Create visualizations for CIFAR-10 results."""
    os.makedirs('figures/cifar10', exist_ok=True)
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'SGD': '#1f77b4', 'Adam': '#9467bd', 'SAM-SGD': '#2ca02c', 'SAM-Adam': '#d62728'}
    
    for opt_name, results in all_results.items():
        history = results['history']
        epochs = range(1, len(history['train_acc']) + 1)
        color = colors.get(opt_name, '#7f7f7f')
        
        # Train accuracy
        axes[0, 0].plot(epochs, history['train_acc'], label=opt_name, color=color, linewidth=2)
        # Test accuracy
        axes[0, 1].plot(epochs, history['test_acc'], label=opt_name, color=color, linewidth=2)
        # Train loss
        axes[1, 0].plot(epochs, history['train_loss'], label=opt_name, color=color, linewidth=2)
        # Generalization gap
        axes[1, 1].plot(epochs, history['generalization_gap'], label=opt_name, color=color, linewidth=2)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Train Accuracy (%)')
    axes[0, 0].set_title('Training Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy (Generalization)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Generalization Gap (%)')
    axes[1, 1].set_title('Generalization Gap (Train - Test)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CIFAR-10 Training Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/cifar10/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/cifar10/training_curves.png")
    
    # 2. Sharpness vs Generalization scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    
    opt_names = list(all_results.keys())
    sharpness = [all_results[name]['sharpness'] for name in opt_names]
    gaps = [all_results[name]['final_generalization_gap'] for name in opt_names]
    test_accs = [all_results[name]['best_test_acc'] for name in opt_names]
    
    for i, name in enumerate(opt_names):
        color = colors.get(name, '#7f7f7f')
        ax.scatter(sharpness[i], gaps[i], s=200, alpha=0.6, color=color, label=name)
        ax.annotate(name, (sharpness[i], gaps[i]), fontsize=9, ha='center', va='bottom')
    
    ax.set_xlabel('Sharpness (Lower is Flatter)', fontsize=11)
    ax.set_ylabel('Generalization Gap (%)', fontsize=11)
    ax.set_title('Sharpness vs Generalization Gap on CIFAR-10', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/cifar10/sharpness_vs_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: figures/cifar10/sharpness_vs_gap.png")


if __name__ == '__main__':
    # Check if PyTorch is available
    try:
        import torch
        import torchvision
        all_results = run_all_cifar10_experiments()
    except ImportError:
        print("PyTorch not installed. Skipping CIFAR-10 experiments.")
        print("Install with: pip install torch torchvision")
