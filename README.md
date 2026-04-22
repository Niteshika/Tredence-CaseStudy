# The Self-Pruning Neural Network
 
A PyTorch implementation of a neural network that learns to prune itself during training, built on the CIFAR-10 dataset.
 
---
 
## Problem Statement
 
Deploying large neural networks is often constrained by memory and computational budgets. This project implements **dynamic self-pruning** — instead of pruning weights after training, the network learns which weights are unnecessary *during* training itself.
 
Each weight is associated with a learnable "gate" parameter (a scalar between 0 and 1). When a gate collapses to zero, the corresponding weight is effectively removed from the network. A sparsity regularization term in the loss function encourages most gates to become exactly zero, leaving only a sparse network of the most important connections.
 
---
 
## Implementation
 
### `PrunableLinear` Layer
 
A custom replacement for `nn.Linear` with three parameters:
- `weight` — standard weight matrix
- `bias` — standard bias
- `gate_scores` — learnable scores, same shape as `weight`, initialized to 2.0 so `sigmoid(2) ≈ 0.88`
**Forward pass:**
```
gates = sigmoid(gate_scores)           # constrain to (0, 1)
pruned_weights = weight * gates        # element-wise masking
output = pruned_weights @ x + bias
```
 
Gradients flow correctly through both `weight` and `gate_scores` via standard autograd.
 
### Sparsity Loss
 
```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```
 
`SparsityLoss` is the **L1 norm** (sum) of all gate values across all `PrunableLinear` layers. L1 is chosen because it applies constant gradient pressure toward zero, causing small gates to collapse completely rather than just shrinking (as L2 would).
 
### Lambda Scheduling
 
Lambda is gradually increased over training using:
 
```python
def get_lambda_schedule(epoch, max_epochs, final_lambda):
    warmup_epochs = int(max_epochs * 0.67)
    if epoch < warmup_epochs:
        return final_lambda * (epoch / warmup_epochs)
    else:
        return final_lambda
```
 
This lets the network learn meaningful representations before sparsity pressure kicks in, preventing premature gate collapse.
 
---
 
## Results
 
| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------|--------------------|
| 0e+00  | 51.84            | 0.00               |
| 1e-06  | 52.30            | 0.10               |
| 1e-05  | 53.42            | 24.63              |
| 3e-05  | 54.21            | 47.91              |
| 7e-05  | 55.09            | 69.29              |
| 1e-04  | 55.45            | 76.44              |
 
**Key takeaway:** Higher λ produces higher sparsity. Accuracy remains competitive even at high sparsity levels, demonstrating that most pruned weights were genuinely redundant.
 
### Why L1 Encourages Sparsity
 
The L1 penalty adds a **constant gradient** of magnitude λ pointing toward zero for every gate, regardless of the gate's current value. This means even very small gates continue to receive the same push toward zero, eventually collapsing completely. In contrast, L2 penalty produces a gradient proportional to the gate's value — small gates receive a tiny push and never fully reach zero. L1 is therefore the natural choice when the goal is exact zeros (true pruning) rather than just small values.
 
---
 
## Gate Distribution Plots
 
Gate distribution histograms for each lambda value are included in the repository:
 
| File | Description |
|------|-------------|
| `gate_dist_lambda_0e+00.png` | No sparsity — gates cluster near 0.88 (initialization) |
| `gate_dist_lambda_1e-06.png` | Weak pressure — distribution shifts slightly left |
| `gate_dist_lambda_1e-05.png` | Moderate pruning — large spike at 0, long tail toward 1 |
| `gate_dist_lambda_3e-05.png` | ~48% sparsity |
| `gate_dist_lambda_7e-05.png` | ~69% sparsity |
| `gate_dist_lambda_1e-04.png` | Aggressive pruning — 76% of gates collapsed to zero |
 
---
 
## How to Run
 
1. Open `notebook.ipynb` in [Kaggle](https://www.kaggle.com) or any Jupyter environment
2. All dependencies (PyTorch, torchvision) are pre-installed on Kaggle
3. Run all cells — CIFAR-10 will be downloaded automatically
4. Training runs for 30 epochs per lambda value
---
 
## Files
 
```
├── notebook.ipynb               # Main implementation
├── gate_dist_lambda_0e+00.png   # Gate distribution plots
├── gate_dist_lambda_1e-06.png
├── gate_dist_lambda_1e-05.png
├── gate_dist_lambda_3e-05.png
├── gate_dist_lambda_7e-05.png
├── gate_dist_lambda_1e-04.png
└── README.md
```
 
---
 
## Dependencies
 
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
 
