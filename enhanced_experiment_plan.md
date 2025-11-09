# Enhanced Experiment Plan: Generalization Gap and Sharpness Analysis

## Research Objective (Enhanced)
To empirically investigate the **generalization gap** phenomenon in optimization algorithms by:
1. Extending comparative analysis to real neural network training
2. Measuring loss landscape sharpness at converged solutions
3. Connecting sharpness to generalization performance
4. Introducing SAM (Sharpness-Aware Minimization) as a modern baseline

## Core Research Questions

### Primary Question
**Why do adaptive methods (Adam) converge faster but often generalize worse than SGD+Momentum?**

### Sub-questions
1. Do adaptive methods converge to sharper minima on both test functions and neural networks?
2. Is there a quantifiable relationship between sharpness and generalization?
3. Can SAM successfully find flatter minima and improve generalization?
4. Are patterns observed on test functions predictive of neural network behavior?

## Enhanced Methodology

### Part 1: Test Functions with Sharpness Analysis (Extended)

**Previous work**: Basic convergence comparison on 4 test functions

**New additions**:
1. **Sharpness Measurement**: For each converged solution, compute sharpness as:
   - Maximum eigenvalue of Hessian matrix
   - Average loss in a small neighborhood (ε-ball)
   - Ratio of sharp/flat regions
   
2. **SAM Implementation**: Add SAM optimizer with ρ=0.05 parameter

3. **Sharpness-Performance Correlation**: Analyze correlation between final sharpness and final loss

### Part 2: Neural Network Experiments (NEW)

**Dataset**: CIFAR-10 (60,000 32×32 color images, 10 classes)

**Architecture**: ResNet-18 (11M parameters)
- Standard architecture with Batch Normalization
- Without BatchNorm variant for comparison

**Training Protocol**:
- Train each optimizer for 200 epochs
- Batch size: 128
- Learning rate schedule: Cosine annealing
- Data augmentation: Random crop, horizontal flip
- 5 independent runs with different random seeds

**Optimizers to Compare**:
1. SGD (lr=0.1)
2. SGD+Momentum (lr=0.1, momentum=0.9)
3. Adam (lr=0.001)
4. AdamW (lr=0.001, weight_decay=0.01)
5. SAM-SGD (lr=0.1, rho=0.05)
6. SAM-Adam (lr=0.001, rho=0.05)

**Metrics to Track**:
- Training loss
- Training accuracy
- Test accuracy (generalization performance)
- Convergence speed (epochs to 90% best accuracy)
- Final generalization gap = Train Acc - Test Acc

**Sharpness Analysis**:
After training, for each converged model:
1. **Hessian-based sharpness**: Compute top eigenvalue of loss Hessian
2. **Neighborhood-based sharpness**: 
   - Sample random perturbations in weight space
   - Measure loss increase: sharpness = max(L(w+δ) - L(w)) / ||δ||²
3. **Filter-wise normalization**: Account for scale invariance

### Part 3: Connecting Test Functions to Neural Networks

**Analysis**:
1. Compare optimizer rankings on test functions vs. neural networks
2. Correlate sharpness metrics across both settings
3. Identify which test function characteristics predict neural network behavior

## Expected Novel Contributions

1. **Empirical Evidence**: First systematic study connecting optimizer behavior on test functions to neural network generalization
2. **Sharpness-Generalization Link**: Quantitative evidence for sharpness-generalization hypothesis
3. **SAM Evaluation**: Comprehensive evaluation of SAM across both settings
4. **Practical Insights**: When to choose Adam vs. SGD vs. SAM based on problem characteristics

## Implementation Plan

### Phase 1: Extend Test Function Experiments
- [x] Implement Hessian eigenvalue computation
- [x] Implement neighborhood-based sharpness measurement
- [x] Add SAM optimizer implementation
- [x] Re-run experiments with sharpness metrics
- [x] Generate new visualizations (sharpness vs. loss plots)

### Phase 2: Neural Network Experiments
- [ ] Set up CIFAR-10 data loading
- [ ] Implement ResNet-18 architecture
- [ ] Implement SAM wrapper for PyTorch optimizers
- [ ] Run training experiments (6 optimizers × 5 runs = 30 runs)
- [ ] Track and save all metrics
- [ ] Compute sharpness for trained models
- [ ] Generate visualizations:
  - Training/test curves
  - Generalization gap comparison
  - Sharpness vs. generalization scatter plots
  - Loss landscape 2D slices around minima

### Phase 3: Updated Paper
- [ ] Expand Introduction with generalization gap motivation
- [ ] Add Sharpness Analysis section in Methodology
- [ ] Add Neural Network Experiments section
- [ ] Add Discussion connecting both experiment types
- [ ] Update Conclusion with novel insights
- [ ] Add new figures and tables
- [ ] Expand references (SAM, sharpness papers)

## Timeline Estimate
- Phase 1: ~30 minutes (computation + implementation)
- Phase 2: ~2-3 hours (neural network training)
- Phase 3: ~30 minutes (paper updates)

Total: ~3-4 hours

## Enhanced Paper Structure

**New Title**: "From Sharp to Flat: An Empirical Study of Optimizer-Induced Loss Landscape Geometry and its Impact on Generalization"

**New Abstract**: Emphasize generalization gap, sharpness analysis, and unified findings

**New Sections**:
1. Introduction: Motivation from generalization gap
2. Related Work: Add sharpness, SAM, generalization literature
3. Methodology: 
   - Test Functions
   - Sharpness Metrics (NEW)
   - Neural Network Protocol (NEW)
4. Experiments:
   - Test Function Results with Sharpness
   - Neural Network Results (NEW)
   - Cross-Domain Analysis (NEW)
5. Discussion: Unified interpretation
6. Conclusion: Novel insights on optimizer selection

This enhanced approach directly addresses the "Why" and "So What" questions, providing novel insights into the fundamental connection between optimization dynamics, loss landscape geometry, and generalization performance.
