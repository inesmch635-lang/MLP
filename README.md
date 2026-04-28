# Full Name:Mecheter Ines 
# MLP Surface Approximation: The Island Explorer 

## Overview
This project implements a **Multi-Layer Perceptron (MLP)** from scratch using **NumPy** to solve a complex regression problem. The goal is to act as an "explorer" who must redraw the topographical map of an island based on scattered elevation data points.

The model learns to approximate the mapping: $f: \mathbb{R}^2 \to \mathbb{R}$, where $(x, y)$ represents coordinates and $z$ represents the altitude.

##  Mathematical Objective
The target altitude function is defined by the following non-linear surface:
$$f(x, y) = \sin(\sqrt{x^2 + y^2}) + 0.5 \cos(2x + 2y)$$

##  Key Technical Features
- **Vectorized Implementation:** Strictly follows matrix calculus constraints. No `for` loops are used for processing samples, ensuring high performance using NumPy.
- **Manual Backpropagation:** Complete manual implementation of gradients and weight updates using the Chain Rule.
- **Customizable Architecture:** Default structure [2, 64, 64, 1] optimized for non-linear surface fitting.
- **Normalization:** Features Min-Max scaling for targets and standard normalization for inputs to ensure gradient stability.

##  Installation & Usage
```
python tp.py
```



1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
