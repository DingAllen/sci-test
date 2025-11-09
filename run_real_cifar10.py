"""
Real CIFAR-10 experiments with actual dataset.
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


# SAM Optimizer wrapper
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
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
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def train_epoch(model, trainloader, criterion, optimizer, device, use_sam=False):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_sam:
            # First forward-backward pass
            def closure():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            
            optimizer.zero_grad()
            loss = closure()
            optimizer.first_step(zero_grad=True)
            
            # Second forward-backward pass
            closure()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return train_loss / (batch_idx + 1), 100. * correct / total


def test(model, testloader, criterion, device):
    """Evaluate on test set."""
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
    
    return test_loss / (batch_idx + 1), 100. * correct / total


def compute_sharpness(model, dataloader, criterion, device, epsilon=0.01, num_samples=10):
    """
    Compute sharpness as maximum loss increase under random perturbations.
    """
    model.eval()
    original_state = {name: param.clone() for name, param in model.named_parameters()}
    
    # Get baseline loss
    baseline_loss = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            baseline_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    baseline_loss /= total
    
    max_sharpness = 0
    
    for _ in range(num_samples):
        # Add random perturbation
        for name, param in model.named_parameters():
            noise = torch.randn_like(param) * epsilon
            param.data.add_(noise)
        
        # Compute perturbed loss
        perturbed_loss = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                perturbed_loss += loss.item() * inputs.size(0)
                total += inputs.size(0)
        perturbed_loss /= total
        
        # Compute sharpness
        sharpness = (perturbed_loss - baseline_loss) / (epsilon ** 2)
        max_sharpness = max(max_sharpness, sharpness)
        
        # Restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(original_state[name])
    
    return max_sharpness


def run_cifar10_experiment(optimizer_name, lr, epochs=100, batch_size=128, seed=42):
    """Run a complete CIFAR-10 experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {optimizer_name} (lr={lr}, epochs={epochs}, seed={seed})")
    print(f"{'='*70}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Data
    print("Loading data...")
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
    
    # Use the converted pickle format data (no download needed)
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    print("Creating model...")
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    use_sam = 'SAM' in optimizer_name
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_name == 'SAM-SGD':
        base_optimizer = lambda params, **kwargs: torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    elif optimizer_name == 'SAM-Adam':
        base_optimizer = lambda params, **kwargs: torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
        optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer if use_sam else optimizer, T_max=epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'gap': [],
        'epochs': list(range(1, epochs + 1))
    }
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, use_sam)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['gap'].append(train_acc - test_acc)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Gap: {train_acc-test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.2f} minutes")
    
    # Compute final sharpness
    print("Computing sharpness...")
    sharpness = compute_sharpness(model, testloader, criterion, device)
    print(f"Sharpness: {sharpness:.6f}")
    
    # Save results
    results = {
        'optimizer': optimizer_name,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'seed': seed,
        'device': device,
        'training_time': elapsed_time,
        'history': history,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'final_gap': train_acc - test_acc,
        'best_test_acc': max(history['test_acc']),
        'sharpness': sharpness,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def run_all_experiments():
    """Run experiments for all optimizers."""
    print("="*80)
    print("REAL CIFAR-10 EXPERIMENTS: GENERALIZATION GAP ANALYSIS")
    print("="*80)
    print(f"\nStarting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration - Use 50 epochs for reasonable training time
    configs = [
        ('SGD', 0.1, 50),
        ('Adam', 0.001, 50),
        ('SAM-SGD', 0.1, 50),
        ('SAM-Adam', 0.001, 50),
    ]
    
    all_results = {}
    
    for opt_name, lr, epochs in configs:
        try:
            results = run_cifar10_experiment(opt_name, lr, epochs=epochs, seed=42)
            all_results[opt_name] = results
            
            # Save individual results
            os.makedirs('results/cifar10_real', exist_ok=True)
            with open(f'results/cifar10_real/{opt_name.lower()}_real.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Saved results for {opt_name}")
            
        except Exception as e:
            print(f"✗ Error with {opt_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if all_results:
        print("\nTest Accuracy Rankings:")
        sorted_by_acc = sorted(all_results.items(), key=lambda x: x[1]['best_test_acc'], reverse=True)
        for rank, (opt_name, results) in enumerate(sorted_by_acc, 1):
            print(f"{rank}. {opt_name}: Test Acc = {results['best_test_acc']:.2f}%, "
                  f"Gap = {results['final_gap']:.2f}%, "
                  f"Sharpness = {results['sharpness']:.6f}")
        
        print("\nSharpness Rankings (Lower is Flatter/Better):")
        sorted_by_sharp = sorted(all_results.items(), key=lambda x: x[1]['sharpness'])
        for rank, (opt_name, results) in enumerate(sorted_by_sharp, 1):
            print(f"{rank}. {opt_name}: Sharpness = {results['sharpness']:.6f}, "
                  f"Test Acc = {results['best_test_acc']:.2f}%")
    
    return all_results


if __name__ == '__main__':
    all_results = run_all_experiments()
