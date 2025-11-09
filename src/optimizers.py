"""
Optimizer implementations for the comparative study.
All optimizers follow a consistent interface for fair comparison.
"""

import numpy as np
from typing import Callable, List, Tuple


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.iteration = 0
    
    def step(self, x, gradient):
        """Perform one optimization step."""
        raise NotImplementedError
    
    def reset(self):
        """Reset optimizer state."""
        self.iteration = 0


class SGD(Optimizer):
    """Stochastic Gradient Descent."""
    
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.name = "SGD"
    
    def step(self, x, gradient):
        self.iteration += 1
        return x - self.learning_rate * gradient
    
    def reset(self):
        super().reset()


class SGDMomentum(Optimizer):
    """SGD with Momentum."""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
        self.name = "SGD-Momentum"
    
    def step(self, x, gradient):
        if self.velocity is None:
            self.velocity = np.zeros_like(x)
        
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
        self.iteration += 1
        return x + self.velocity
    
    def reset(self):
        super().reset()
        self.velocity = None


class AdaGrad(Optimizer):
    """AdaGrad optimizer."""
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.sum_squared_gradients = None
        self.name = "AdaGrad"
    
    def step(self, x, gradient):
        if self.sum_squared_gradients is None:
            self.sum_squared_gradients = np.zeros_like(x)
        
        self.sum_squared_gradients += gradient ** 2
        adjusted_gradient = gradient / (np.sqrt(self.sum_squared_gradients) + self.epsilon)
        self.iteration += 1
        return x - self.learning_rate * adjusted_gradient
    
    def reset(self):
        super().reset()
        self.sum_squared_gradients = None


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.squared_gradients = None
        self.name = "RMSprop"
    
    def step(self, x, gradient):
        if self.squared_gradients is None:
            self.squared_gradients = np.zeros_like(x)
        
        self.squared_gradients = (self.decay_rate * self.squared_gradients + 
                                  (1 - self.decay_rate) * gradient ** 2)
        adjusted_gradient = gradient / (np.sqrt(self.squared_gradients) + self.epsilon)
        self.iteration += 1
        return x - self.learning_rate * adjusted_gradient
    
    def reset(self):
        super().reset()
        self.squared_gradients = None


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.name = "Adam"
    
    def step(self, x, gradient):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.iteration += 1
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)
        
        # Update parameters
        return x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""
    
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None  # First moment
        self.v = None  # Second moment
        self.name = "AdamW"
    
    def step(self, x, gradient):
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.iteration += 1
        
        # Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
        
        # Compute bias-corrected moment estimates
        m_hat = self.m / (1 - self.beta1 ** self.iteration)
        v_hat = self.v / (1 - self.beta2 ** self.iteration)
        
        # Update parameters with decoupled weight decay
        x_new = x - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + 
                                           self.weight_decay * x)
        return x_new
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None


def get_optimizer(name, learning_rate=0.01, **kwargs):
    """Factory function to create optimizer by name."""
    optimizers = {
        'sgd': SGD,
        'sgd-momentum': SGDMomentum,
        'adagrad': AdaGrad,
        'rmsprop': RMSprop,
        'adam': Adam,
        'adamw': AdamW,
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name.lower()](learning_rate=learning_rate, **kwargs)
