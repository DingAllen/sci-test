"""
Quick CIFAR-10 experiments - trains for fewer epochs to demonstrate key phenomena.
This runs actual training on real CIFAR-10 data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import pickle


# Simplified ResNet for faster training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""
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
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
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


def load_cifar10():
    """Load CIFAR-10 from local files."""
    print("Loading CIFAR-10...")
    data_dir = 'data/cifar-10-batches-py'
    
    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            train_data.append(batch[b'data'])
            train_labels.extend(batch[b'labels'])
    
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).astype(np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)
    
    # Load test data
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
        test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32)
        test_labels = np.array(test_batch[b'labels'], dtype=np.int64)
    
    # Normalize (CIFAR-10 standard normalization)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    
    train_data = (train_data / 255.0 - mean) / std
    test_data = (test_data / 255.0 - mean) / std
    
    print(f"  Train: {train_data.shape}, Test: {test_data.shape}")
    return train_data, train_labels, test_data, test_labels


def train_epoch(model, train_X, train_y, optimizer, criterion, device, batch_size, is_sam):
    """Train for one epoch."""
    model.train()
    indices = np.random.permutation(len(train_X))
    total_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:min(i+batch_size, len(indices))]
        X_batch = torch.FloatTensor(train_X[batch_idx]).to(device)
        y_batch = torch.LongTensor(train_y[batch_idx]).to(device)
        
        if is_sam:
            # SAM requires two forward-backward passes
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            criterion(model(X_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * len(batch_idx)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return 100. * correct / total, total_loss / total


def test(model, test_X, test_y, criterion, device, batch_size):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(test_X), batch_size):
            X_batch = torch.FloatTensor(test_X[i:i+batch_size]).to(device)
            y_batch = torch.LongTensor(test_y[i:i+batch_size]).to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return 100. * correct / total, total_loss / total


def compute_sharpness(model, test_X, test_y, criterion, device, batch_size, rho=0.05):
    """Compute sharpness metric."""
    model.eval()
    
    # Base loss
    base_loss, _ = test(model, test_X, test_y, criterion, device, batch_size)
    
    # Perturb weights
    original_weights = {name: param.data.clone() for name, param in model.named_parameters()}
    
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * rho * param.abs().mean()
            param.add_(noise)
    
    # Perturbed loss
    perturbed_loss, _ = test(model, test_X, test_y, criterion, device, batch_size)
    
    # Restore weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.data.copy_(original_weights[name])
    
    sharpness = (perturbed_loss - base_loss) / (abs(base_loss) + 1e-8)
    return max(0, sharpness)


def train_optimizer(opt_name, train_X, train_y, test_X, test_y, epochs=20, batch_size=256, lr=0.01):
    """Train with specified optimizer."""
    print(f"\n{'='*60}")
    print(f"Training with {opt_name}")
    print(f"{'='*60}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Create model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    is_sam = 'SAM' in opt_name
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    elif opt_name == 'SAM-SGD':
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=0.9, weight_decay=5e-4, rho=0.05)
    elif opt_name == 'SAM-Adam':
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=0.001, weight_decay=5e-4, rho=0.05)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    # Training
    history = {'epochs': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'gap': []}
    start_time = time.time()
    
    for epoch in range(epochs):
        train_acc, train_loss = train_epoch(model, train_X, train_y, optimizer, criterion, device, batch_size, is_sam)
        test_acc, test_loss = test(model, test_X, test_y, criterion, device, batch_size)
        
        history['epochs'].append(epoch)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['gap'].append(train_acc - test_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train {train_acc:.2f}%, Test {test_acc:.2f}%, Gap {train_acc-test_acc:.2f}%")
    
    # Compute sharpness
    print("Computing sharpness...")
    sharpness = compute_sharpness(model, test_X, test_y, criterion, device, batch_size)
    
    training_time = time.time() - start_time
    
    results = {
        'optimizer': opt_name,
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'final_gap': history['gap'][-1],
        'sharpness': sharpness,
        'history': history,
        'training_time': training_time
    }
    
    print(f"\nFinal: Train {results['final_train_acc']:.2f}%, Test {results['final_test_acc']:.2f}%, "
          f"Gap {results['final_gap']:.2f}%, Sharpness {sharpness:.4f}, Time {training_time:.1f}s")
    
    return results


def main():
    """Run all experiments."""
    print("="*60)
    print("REAL CIFAR-10 EXPERIMENTS")
    print("="*60)
    
    # Load data
    train_X, train_y, test_X, test_y = load_cifar10()
    
    # Create output directory
    os.makedirs('results/cifar10_real', exist_ok=True)
    
    # Run experiments
    optimizers = ['SGD', 'Adam', 'SAM-SGD', 'SAM-Adam']
    epochs = 30  # Good balance of demonstration vs. time
    
    all_results = {}
    for opt_name in optimizers:
        results = train_optimizer(opt_name, train_X, train_y, test_X, test_y, epochs=epochs)
        all_results[opt_name] = results
        
        # Save results
        filename = f'results/cifar10_real/{opt_name.lower().replace("-", "_")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"→ Saved to {filename}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Optimizer':<12} {'Train%':<8} {'Test%':<8} {'Gap%':<8} {'Sharp':<8} {'Time(s)':<8}")
    print("-"*60)
    for opt_name in optimizers:
        r = all_results[opt_name]
        print(f"{opt_name:<12} {r['final_train_acc']:>6.2f}% {r['final_test_acc']:>6.2f}% "
              f"{r['final_gap']:>6.2f}% {r['sharpness']:>6.4f} {r['training_time']:>7.1f}")
    
    print("="*60)
    print("✓ All experiments completed!")
    print(f"✓ Results saved to: results/cifar10_real/")


if __name__ == '__main__':
    main()
