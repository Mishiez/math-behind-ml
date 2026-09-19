# Math Behind ML

This repository explores the mathematical foundations behind machine learning and how those concepts are used during model training.

The work begins with understanding the underlying concepts before implementing them from scratch. The goal is to understand what is happening inside a machine learning model rather than relying only on high-level `.fit()` methods.

## Concepts

### 1. Parameters

Parameters are values inside a machine learning model that can be adjusted during training.

For a simple model:

$$
y = wx + b
$$

* `x` is the input.
* `w` is the weight.
* `b` is the bias.
* `y` is the prediction.

The model learns by adjusting its parameters, such as the weights and biases.

---

### 2. Loss

A model needs a way to measure how wrong its predictions are.

A **loss function** produces a numerical value representing the error between the model's prediction and the correct answer.

* High loss → the prediction is more wrong.
* Low loss → the prediction is closer to the correct answer.

Training a model involves trying to reduce this loss.

---

### 3. Derivatives

A derivative describes how a value changes when another value changes.

In machine learning, derivatives help us understand how changing a model parameter affects the loss.

For example, if changing a weight causes the loss to increase, the derivative tells us that the loss is increasing in that direction.

This information helps determine how the model's parameters should be adjusted.

---

### 4. Gradient

A model usually has many parameters rather than just one.

A derivative tells us how the loss changes with respect to one parameter, while a **gradient** contains this information for all the parameters.

The gradient therefore tells us the direction in which the loss changes most strongly with respect to the model's parameters.

---

### 5. Gradient Descent

Gradient descent is an optimization method used to reduce the loss.

The basic idea is to:

1. Calculate the current loss.
2. Calculate the gradient.
3. Move the parameters in the direction that reduces the loss.
4. Repeat the process.

The basic update is:

$$
\theta := \theta - \alpha \nabla L
$$

Where:

* `θ` is a model parameter.
* `α` is the learning rate.
* `∇L` is the gradient of the loss.

The learning rate controls how large each update is.

If the updates are too small, learning can be slow. If they are too large, the model can overshoot the minimum instead of converging toward it.

---

### 6. Convergence

During training, we want the loss to decrease.

For example:

```text
High loss
   ↓
   ↓
   ↓
   ↓
Low loss
```

When the loss and parameters eventually stop changing significantly, the optimization process is said to have **converged**.

A loss curve can be used to visualize this process.

---

### 7. Neural Network Forward Pass

A neural network contains layers of parameters that transform an input into a prediction.

A simple two-layer network can be represented as:

```text
Input
  ↓
Layer 1
  ↓
Activation
  ↓
Layer 2
  ↓
Prediction
  ↓
Loss
```

The forward pass is the process of moving from the input through the network until a prediction is produced.

For example, a layer may calculate:

$$
Z = XW + b
$$

and then apply an activation function to produce the output passed to the next layer.

---

### 8. Chain Rule

The chain rule allows us to calculate how a change at the beginning of a sequence of operations affects the final result.

For example:

```text
A → B → C → Loss
```

If we want to understand how changing `A` affects the final loss, we can trace the effect through each intermediate step.

Conceptually:

$$
\frac{dL}{dA}
=
\frac{dL}{dC}
\frac{dC}{dB}
\frac{dB}{dA}
$$

The chain rule is particularly important for neural networks because a parameter can affect the final loss through several layers of calculations.

---

### 9. Backpropagation

Backpropagation uses the chain rule to calculate how each parameter in a neural network contributed to the final loss.

The forward pass moves through the network:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Prediction
  ↓
Loss
```

Backpropagation works backward:

```text
Loss
  ↓
Layer 2 gradients
  ↓
Layer 1 gradients
  ↓
Parameter gradients
```

The resulting gradients can then be used by gradient descent to update the model's parameters.

---

## How the Concepts Connect

The overall training process can be viewed as:

```text
                FORWARD PASS
                     ↓
Input → Model → Prediction
                     ↓
                   Loss
                     ↓
               BACKPROPAGATION
                     ↓
                  Gradients
                     ↓
               GRADIENT DESCENT
                     ↓
              Updated Parameters
                     ↓
                   Repeat
```

In simple terms:

* **Loss** tells us how wrong the model is.
* **Derivatives** tell us how the loss changes.
* **Gradients** collect this information for the model's parameters.
* **Backpropagation** calculates those gradients by working backward through the network.
* **Gradient descent** uses the gradients to update the parameters.
* Repeating this process allows the model to learn.

## Implementation

The practical work in this repository implements:

1. Gradient descent from scratch using NumPy — [`gradient_descent.ipynb`](gradient_descent.ipynb).
2. A simple loss function and a converging loss curve — same notebook.
3. A two-layer neural network from scratch, including the paper derivation — [`two_layer_nn.ipynb`](two_layer_nn.ipynb).
4. Forward propagation.
5. Backpropagation using manually derived gradients.
6. Parameter updates and a training loop using gradient descent.
7. Numerical comparison against an equivalent reference implementation.

The from-scratch implementation does not use automatic differentiation or a machine learning framework to calculate the gradients — only NumPy. PyTorch is used only as an external reference to validate the results, in a clearly separated comparison section.

## Results and Validation

### Gradient descent (`gradient_descent.ipynb`)

Minimising $L(w) = (w-3)^2$ from $w_0 = 0$ with learning rate $\alpha = 0.1$ for 50 iterations:

* Final $w$: 2.999957 (target: 3)
* Final loss: $1.83 \times 10^{-9}$ (target: 0)

Comparing four learning rates confirms the expected behaviour of gradient descent on this loss:

| Learning rate | Behaviour | Outcome |
|---|---|---|
| 0.01 | too small | converges, but very slowly |
| 0.10 | good | converges smoothly |
| 0.90 | oscillates | converges, but overshoots the minimum every step |
| 1.10 | too large | diverges |

### Two-layer network (`two_layer_nn.ipynb`)

Trained on a standardised `make_moons` toy dataset ($m=200$, $d=2$, $h=4$ hidden units), full-batch gradient descent, learning rate 0.5, 3000 epochs.

**From scratch:**
* Initial loss: 0.6642 — close to $\ln 2 \approx 0.6931$, as expected when small random weights push every prediction near 0.5.
* Final loss: 0.0586
* Final accuracy: 96.5%
* Loss decreases monotonically, has no NaNs, and stabilises over the last 500 epochs (checked explicitly in the notebook, not just eyeballed off the plot).

**Numerical gradient check** (central difference, $\epsilon = 10^{-5}$, checked on every one of the 17 parameters):
* Max relative error vs. the analytic `backward` gradients: ~$4 \times 10^{-9}$
* Tolerance: $10^{-5}$ — passed for every parameter.

**Reference comparison (PyTorch)** — same architecture, same exact initial weights, same learning rate, plain full-batch SGD, float64:
* Gradients at initialisation (autograd vs. analytic `backward`): max difference ~$10^{-17}$.
* After training: final loss, predictions, and all four weight matrices match to within ~$10^{-15}$.

Tolerance used throughout the reference comparison: $10^{-5}$. Every comparison landed several orders of magnitude inside it — the remaining differences are floating-point noise, not implementation error.

## Goal

The goal is not simply to produce a working implementation.

The goal is to understand **what is happening inside the model during training**, including how the loss produces gradients, how backpropagation applies the chain rule, and how gradient descent uses those gradients to improve the model.
