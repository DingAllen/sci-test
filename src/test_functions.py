"""
Test functions module for optimization experiments.
Implements various non-convex test functions commonly used in optimization research.
"""

import numpy as np


class TestFunction:
    """Base class for test functions."""
    
    def __init__(self, dim=2):
        self.dim = dim
        self.bounds = None
        self.global_optimum = None
        self.optimal_value = None
    
    def __call__(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        """Compute gradient using finite differences."""
        epsilon = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * epsilon)
        return grad


class RastriginFunction(TestFunction):
    """
    Rastrigin function: highly multi-modal with many local minima.
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0,...,0) = 0
    """
    
    def __init__(self, dim=2):
        super().__init__(dim)
        self.bounds = (-5.12, 5.12)
        self.global_optimum = np.zeros(dim)
        self.optimal_value = 0.0
    
    def __call__(self, x):
        x = np.asarray(x)
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class AckleyFunction(TestFunction):
    """
    Ackley function: nearly flat outer region with a large hole at the center.
    Global minimum: f(0,...,0) = 0
    """
    
    def __init__(self, dim=2):
        super().__init__(dim)
        self.bounds = (-5.0, 5.0)
        self.global_optimum = np.zeros(dim)
        self.optimal_value = 0.0
    
    def __call__(self, x):
        x = np.asarray(x)
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - 
                np.exp(sum_cos / n) + 20 + np.e)


class RosenbrockFunction(TestFunction):
    """
    Rosenbrock function: valley-shaped, challenging for optimization.
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1,...,1) = 0
    """
    
    def __init__(self, dim=2):
        super().__init__(dim)
        self.bounds = (-2.5, 2.5)
        self.global_optimum = np.ones(dim)
        self.optimal_value = 0.0
    
    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class BealeFunction(TestFunction):
    """
    Beale function: narrow valley optimization problem.
    Global minimum: f(3, 0.5) = 0
    Only defined for 2D.
    """
    
    def __init__(self):
        super().__init__(dim=2)
        self.bounds = (-4.5, 4.5)
        self.global_optimum = np.array([3.0, 0.5])
        self.optimal_value = 0.0
    
    def __call__(self, x):
        x = np.asarray(x)
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1 * x2)**2
        term2 = (2.25 - x1 + x1 * x2**2)**2
        term3 = (2.625 - x1 + x1 * x2**3)**2
        return term1 + term2 + term3


class SphereFunction(TestFunction):
    """
    Sphere function: simple convex function for baseline comparison.
    f(x) = sum(x_i^2)
    Global minimum: f(0,...,0) = 0
    """
    
    def __init__(self, dim=2):
        super().__init__(dim)
        self.bounds = (-5.0, 5.0)
        self.global_optimum = np.zeros(dim)
        self.optimal_value = 0.0
    
    def __call__(self, x):
        x = np.asarray(x)
        return np.sum(x**2)


def get_test_function(name, dim=2):
    """Factory function to get test function by name."""
    functions = {
        'rastrigin': RastriginFunction,
        'ackley': AckleyFunction,
        'rosenbrock': RosenbrockFunction,
        'beale': BealeFunction,
        'sphere': SphereFunction,
    }
    
    if name.lower() not in functions:
        raise ValueError(f"Unknown function: {name}")
    
    func_class = functions[name.lower()]
    if name.lower() == 'beale':
        return func_class()
    else:
        return func_class(dim)
