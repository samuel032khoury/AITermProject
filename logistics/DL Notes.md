## Overview

- Structured Data Vs. Unstructured Data
- Scale drives DL
  - Amount of Data
  - Size of NN (hidden nodes, connections)
- Change sigmoid function to ReLU function makes gradient descent work much faster

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

## Neural Networks

Networks made of nodes each representing a logistic regression.

Goal: find $w^{[i]}$ and $b^{[i]}$ to minimize loss

Notation: Round bracket refers to i-th training instances; Square bracket refers to the layer of the network.

Activation Variable: $a^{[i]}$

### Structure

[Input Layer] -> [Hidden Layer] ->[Output Layer]

### Growing the Network (Forward Propagation)

Single instance:Single feature:Single Node:Single layer -> Compute $z = w x+b$ and $a = \sigma(z)$. $w\in \R, b\in \R$ `(w.shape = [1,1], b.shape = [1,1])`

Single instance:Multiple features:Single Node:Single layer -> Compute $z = wx^{[0]}+b$ and $a = \sigma(z)$. $w \in \R^{1\times n_x}, b \in \R$ `(w.shape = [1,n], b.shape = [1,1])`

Single instance:Multiple features:Multiple Nodes:Single layer -> Compute $z = Wx^{[0]}+B$ and $a = \sigma(z)$. $(W\in \R^{m\times n_X}, B \in \R^{m\times1})$ `(W.shape = [m,n], B.shape = [m,1])`

Single instance:Multiple features:Multiple Nodes:Multiple layers -> Compute $z^{[k]} = W^{[k]}x^{[k]}+B^{[k]}$ and $a^{[k]} = g^{[k]}(z^{[k]})$.

Multiple instances:Multiple features:Multiple Nodes:Multiple layers -> Compute $Z^{[k]} = W^{[k]} X^{[k]}+B^{[k]}$ and $A^{[k]} = g^{[k]}(Z^{[k]})$

#### Activation Function

- TanH
  - Has codomain of (-1,1), is most often superior to sigmoid

- SoftMax
    > In the two-class logistic regression, the predicted probabilities are as follows, using the sigmoid function:
    > $$
    > \begin{align}
    > \Pr(Y_i=0) &= \frac{e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} \, \\
    > \Pr(Y_i=1) &= 1 - \Pr(Y_i=0) = \frac{1} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}}
    > \end{align}
    > $$
    > In the multi-class logistic regression, with 𝐾K classes, the predicted probabilities are as follows, using the SoftMax function:
    > $$
    > \begin{align}
    > \Pr(Y_i=k) &= \frac{e^{\boldsymbol\beta_k \cdot \mathbf{X}_i}} {~\sum_{0 \leq c \leq K}^{}{e^{\boldsymbol\beta_c \cdot \mathbf{X}_i}}} \, \\
    > \end{align}
    > $$

    > - If you have a multi-label classification problem = there is more than one "right answer" = the outputs are NOT mutually exclusive, then use a sigmoid function on each raw output independently. The sigmoid will allow you to have high probability for all of your classes, some of them, or none of them
    > - If you have a multi-class classification problem = there is only one "right answer" = the outputs are mutually exclusive, then use a SoftMax function. The SoftMax will enforce that the sum of the probabilities of your output classes are equal to one, so in order to increase the probability of a particular class, your model must correspondingly decrease the probability of at least one of the other classes.

- ReLU
  - $a = \max(0,z)$

### Gradient Descent(Backward Propagation)

- Suppose a single-instance, multiple-feature, multiple-node, and two-layer network (one hidden, one output-> $g^{[2]} = \sigma$) with loss function $\mathcal L (a,y) = -y \log a- (1-y)\log(1-a)$
  - $\delta a^{[2]} = \frac\delta{\delta a^{[2]}}\mathcal L(a^{[2]},y) = -\frac y {a^{[2]}} + \frac{1-y}{1-a^{[2]}}$ ($\delta a$ is often omittied in practice)
  - $\delta z^{[2]} = \frac{\delta}{\delta z^{[2]}}\mathcal{L}= \frac{\delta\mathcal L}{\delta a^{[2]}}\cdot \frac{\delta a^{[2]}}{\delta z^{[2]}}= \frac{\delta\mathcal L}{\delta a^{[2]}}\cdot g^{[2]\prime} = a^{[2]} - y$
  - $\delta W^{[2]} = \delta z^{[2]} \cdot a^{[1]T}$
  - $\delta b^{[2]} = \delta z^{[2]}$
  - $\delta z^{[1]}=W^{[2] T} \delta z^{[2]} \cdot g^{[1] \prime}\left(z^{[1]}\right)$
  - $\delta W^{[1]}=\delta z^{[1]} x^{T}$
  - $\delta b^{[1]}=\delta z^{[1]}$
- If the same network but has multiple training instances
  - 
- In neural network, a variable and its backward derivative have the same dimension

### Random Initialization

- Weights need to be randomly initialized to mitigate symmetry breaking.
- $b$ can be initialzied as 0's

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