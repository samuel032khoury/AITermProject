## Overview

- Structured Data Vs. Unstructured Data
- Scale drives DL
  - Amount of Data
  - Size of NN (hidden nodes, connections)
- Change sigmoid function to ReLU function makes gradient descent work much faster

---

## Logistic Regression (Binary Classification)

- find $\omega$ and $b$ such that $\Pr(y = 1|x) = \hat y = \sigma(\omega^T x + b)$, where $\sigma(z) = \frac1{1 + e^{-z}}$, with the codomain of $(0,1)$.

- Loss(error) function [need the value as small as possible - think about LSF]: $L(\hat y , y) = - (y \log \hat y + (1- y)\log(1- \hat y))$
  - In general, the loss function need to be a convex function so it can have a global minima.
- Cost function: $\displaystyle J(\omega, b) = \frac1 m \sum ^m_{i = 1}L({\hat y }^{(i)}, y ^{(i)})$
  - When trining, we need to find $\omega, b$ that minimize $J$.

### Gradient Descent

- $\omega = \omega - \alpha[\frac{\mathrm{\delta}}{\mathrm{\delta\  }\omega}J(\omega, b)]$

- Convention: d$var$ is used to represent the derivative value of the final output variable wrt 'var'.

### Vectorization

- Use vectorization to improve algorithm efficiency
- Vector multiplication
- Matrix multiplication
- derivative updates

---

## Neural Networks

Networks made of nodes each representing a logistic regression.

Goal: find $w^{[i]}$ and $b^{[i]}$ to minimize loss

Notation: Round bracket refers to i-th training instances; Square bracket refers to the layer of the network.

Activation Variable: $a^{[i]}$

### Growing the Network

Single instance:Single feature:Single Node:Single layer

Single instance:Multiple features:Single Node:Single layer

Single instance:Multiple features:Multiple Nodes:Single layer

Single instance:Multiple features:Multiple Nodes:Multiple layers

Multiple instances:Multiple features:Multiple Nodes:Multiple layers

### Structure

[Input Layer] -> [Hidden Layer] ->[Output Layer]

---

## Orthogonalization

Separate variables to represent different features.

### Chain of Assumptions in ML

- Fit training set well on cost function
  - IFNOT: Build a bigger NN; Switch to a better optimization algorithm.
- Fur Dev set well on cost function
  - IFNOT: Get a bigger training set.
- Fit test set well on cost function
  - IFNOT: Get a bigger dev set.
- Performs well in real world
  - IFNOT: Change the dev set or the cost function.