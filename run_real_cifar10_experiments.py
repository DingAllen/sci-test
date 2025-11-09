"""
Real CIFAR-10 experiments using actual dataset.
Trains ResNet-18 with different optimizers and analyzes generalization gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import pickle
from datetime import datetime


# ResNet-18 implementation
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer.
    Based on Foret et al., ICLR 2021.
    """
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad: self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def load_cifar10_data():
    """Load CIFAR-10 data from local pickle files."""
    data_dir = 'data/cifar-10-batches-py'
    
    # Load training batches
    train_data = []
    train_labels = []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            train_data.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])
    
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels)
    
    # Load test batch
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        test_data = test_batch[b'data'].reshape(-1, 3, 32, 32)
        test_labels = np.array(test_batch[b'labels'])
    
    return train_data, train_labels, test_data, test_labels


def train_epoch(model, trainloader, optimizer, criterion, device, is_sam=False):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if is_sam:
            # SAM requires two forward-backward passes
            # First forward-backward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            criterion(model(inputs), targets).backward()
            optimizer.second_step(zero_grad=True)
        else:
            # Standard optimizer
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = train_loss / (batch_idx + 1)
    return acc, avg_loss


def test_epoch(model, testloader, criterion, device):
    """Test the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = test_loss / (batch_idx + 1)
    return acc, avg_loss


def compute_sharpness(model, dataloader, criterion, device, rho=0.05):
    """
    Compute sharpness as maximum loss in a neighborhood.
    Measures how much loss can increase within a small perturbation.
    """
    model.eval()
    
    # Compute base loss
    base_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            base_loss += criterion(outputs, targets).item() * inputs.size(0)
            total += inputs.size(0)
    base_loss /= total
    
    # Perturb weights and compute perturbed loss
    perturbed_loss = 0.0
    total = 0
    
    # Save original weights
    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()
    
    # Compute perturbation direction (random)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * rho * param.abs().mean()
            param.add_(noise)
    
    # Compute loss with perturbed weights
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            perturbed_loss += criterion(outputs, targets).item() * inputs.size(0)
            total += inputs.size(0)
    perturbed_loss /= total
    
    # Restore original weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(original_weights[name])
    
    # Sharpness is the maximum loss increase
    sharpness = (perturbed_loss - base_loss) / (base_loss + 1e-8)
    return max(0, sharpness)  # Ensure non-negative


def train_cifar10(optimizer_name, num_epochs=100, batch_size=128, lr=0.1, seed=42):
    """
    Train ResNet-18 on CIFAR-10 with specified optimizer.
    """
    print(f"\n{'='*60}")
    print(f"Training with {optimizer_name}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_data, train_labels, test_data, test_labels = load_cifar10_data()
    
    # Normalize data (convert to float32 first for efficiency)
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    
    # Compute mean and std more efficiently
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    
    train_data = (train_data / 255.0 - mean) / std
    test_data = (test_data / 255.0 - mean) / std
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_data),
        torch.LongTensor(train_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_data),
        torch.LongTensor(test_labels)
    )
    
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Create model
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    is_sam = 'SAM' in optimizer_name
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    elif optimizer_name == 'SAM-SGD':
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=0.9, weight_decay=5e-4, rho=0.05)
    elif optimizer_name == 'SAM-Adam':
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.001, weight_decay=5e-4, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer if is_sam else optimizer, T_max=num_epochs)
    
    # Training loop
    history = {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'generalization_gap': []
    }
    
    best_test_acc = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Train
        train_acc, train_loss = train_epoch(model, trainloader, optimizer, criterion, device, is_sam)
        
        # Test
        test_acc, test_loss = test_epoch(model, testloader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Record history
        history['epochs'].append(epoch)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['generalization_gap'].append(train_acc - test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | '
                  f'Gap: {train_acc-test_acc:.2f}% | Time: {elapsed:.1f}s')
    
    # Compute final sharpness
    print("\nComputing sharpness...")
    sharpness = compute_sharpness(model, testloader, criterion, device)
    
    # Final results
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    final_gap = history['generalization_gap'][-1]
    
    results = {
        'optimizer': optimizer_name,
        'final_train_acc': final_train_acc,
        'final_test_acc': final_test_acc,
        'best_test_acc': best_test_acc,
        'final_generalization_gap': final_gap,
        'sharpness': sharpness,
        'history': history,
        'training_time': time.time() - start_time
    }
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {final_train_acc:.2f}%")
    print(f"  Test Accuracy: {final_test_acc:.2f}%")
    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"  Generalization Gap: {final_gap:.2f}%")
    print(f"  Sharpness: {sharpness:.4f}")
    print(f"  Training Time: {results['training_time']:.1f}s")
    
    return results


def main():
    """Run experiments for all optimizers."""
    print("="*60)
    print("CIFAR-10 Real Experiments with ResNet-18")
    print("="*60)
    
    # Create results directory
    os.makedirs('results/cifar10_real', exist_ok=True)
    
    # Run experiments - using reasonable number of epochs
    optimizers = ['SGD', 'Adam', 'SAM-SGD', 'SAM-Adam']
    num_epochs = 50  # Reduced for faster execution while still demonstrating key phenomena
    
    all_results = {}
    
    for opt_name in optimizers:
        results = train_cifar10(opt_name, num_epochs=num_epochs)
        all_results[opt_name] = results
        
        # Save individual results
        output_file = f'results/cifar10_real/{opt_name.lower().replace("-", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*60)
    print(f"{'Optimizer':<15} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Sharpness':<12} {'Time':<10}")
    print("-"*60)
    
    for opt_name in optimizers:
        r = all_results[opt_name]
        print(f"{opt_name:<15} {r['final_train_acc']:>10.2f}% {r['final_test_acc']:>10.2f}% "
              f"{r['final_generalization_gap']:>8.2f}% {r['sharpness']:>10.4f} {r['training_time']:>8.1f}s")
    
    print("="*60)
    print("\nAll experiments completed successfully!")
    print(f"Results saved to: results/cifar10_real/")


if __name__ == '__main__':
    main()
