# go-mHC: Direct Parameterization of Manifold-Constrained Hyper-Connections

This repository contains the official implementation of **go-mHC** (*generalized orthostochastic Manifold-Constrained Hyper-Connections*), as presented in the 2026 paper: ["go-mHC: Direct Parameterization of Manifold-Constrained Hyper-Connections via Generalized Orthostochastic Matrices"](https://arxiv.org/abs/2604.02309v1).

## Overview

Learning how to mix information across residual streams is a powerful way to optimize neural architecture connectivity. However, parameterizing the **Birkhoff polytope** (the set of doubly stochastic matrices) exactly and efficiently is challenging. 

**go-mHC** provides a solution grounded in the theory of **generalized orthostochastic matrices**. It scales at `O(d³)`—where `d` is the number of streams—avoiding the factorial scaling of prior exact methods while offering significantly more expressivity than Kronecker-factorized approaches.

### Key Advantages
* **Exact Constraints:** Unlike the original mHC which relies on iterative Sinkhorn-Knopp normalization (and thus only approximates double stochasticity), go-mHC is an **exact parameterization** that satisfies the constraints by construction.
* **Efficiency:** It avoids the **factorial scaling** of mHC-lite ($d!$), instead scaling at $\mathcal{O}(d^3)$ in both parameter count and FLOPs.
* **Tunable Expressivity:** It introduces a single hyperparameter $s$ that allows for a **continuous interpolation** between a computationally efficient boundary ($s=1$) and the fully expressive Birkhoff polytope ($s \to \infty$).
* **Expressivity / Spectral Coverage:** Spectral analysis demonstrates that go-mHC fills the Birkhoff polytope much more than Kronecker-factorized baselines.
* **Convergence Speed:** In synthetic stream-mixing tasks, the method achieves the minimum theoretical loss while converging up to **10x faster** than existing baselines.
* **Simplifies Implementation:** It requires **no custom CUDA kernels** and is implemented using standard, batched linear algebra operations available in common deep learning frameworks.

---

## Repository Structure

```
.
├── tiny_language_model/    # GPT-style model validation experiments (30M params)
│   ├── config/             # Training configs on TinyStories/Shakespeare for HC, mHC, mHC-lite, go-mHC, and KromHC
│   ├── model.py            # Core architecture with go-mHC integration
│   ├── train.py            # Training script
│   └── sample.py           # Generate samples for model evaluation
├── toy_models/             # Research & Analysis notebooks for main paper results
│   ├── dynamics.ipynb      # Training dynamics and convergence speed tests
│   └── spectra.ipynb       # Spectral analysis of the parameterization
└── LICENSE
└── README.md
```

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies (PyTorch is required):
```sh
git clone https://github.com/itstorque/go-mHC.git
cd go-mHC
```

### 3. Reproducing main results
- Spectra and Expressivity: `toy_models/spectra.ipynb`
- Convergence, Loss Trajectories, etc.: `toy_models/dynamics.ipynb`

### 3. Validating the Tiny Language Model
To train the 30M parameter GPT model using the **go-mHC** parameterization on the Shakespeare dataset:
```sh
pip3 install tiktoken datasets tensorboard einops
python tiny_language_model/train.py tiny_language_model/config/with_go_mhc.py
```

---

## Mathematical Procedure

The **go-mHC** pipeline maps a set of learnable parameters to a doubly stochastic matrix $B$ via:
1. **Skew-Symmetric Mapping:** Parameters are mapped to a $ds \times ds$ skew-symmetric matrix.
2. **Cayley Transform:** This generates an orthogonal matrix $Q$.
3. **Frobenius Projection:** We compute $B_{ij} = \frac{1}{s}\|Q_{ij}\|_F^2$ to produce the final $d \times d$ matrix.

We propose using **$s=2$** for the best balance of performance and expressivity, and $s=1$ if expressivity is not needed. This can be used simultaneously with KromHC to improve the expressivity of KromHC with negligible cost.

---

## Citation

If you find this work useful, please cite our paper:

```
@misc{dandachi2026gomhc,
      title={go-$m$HC: Direct Parameterization of Manifold-Constrained Hyper-Connections via Generalized Orthostochastic Matrices}, 
      author={Torque Dandachi and Sophia Diggs-Galligan},
      year={2026},
      eprint={2604.02309},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.02309}, 
}
```