# Neural Networks Intro

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

- Convention: $\text{d }var$ is used to represent the derivative value of the final output variable wrt 'var'.

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

Single instance:Single feature:Single Node:Single layer -> Compute $z = w x+b$ and $a = \sigma(z)$. $w\in \mathbb{R}, b\in \mathbb{R}$ `(w.shape = [1,1], b.shape = [1,1])`

Single instance:Multiple features:Single Node:Single layer -> Compute $z = wx^{[0]}+b$ and $a = \sigma(z)$. $w \in \mathbb{R}^{1\times n_x}, b \in \mathbb{R}$ `(w.shape = [1,n], b.shape = [1,1])`

Single instance:Multiple features:Multiple Nodes:Single layer -> Compute $z = Wx^{[0]}+B$ and $a = \sigma(z)$. $(W\in \mathbb{R}^{m\times n_X}, B \in \mathbb{R}^{m\times1})$ `(W.shape = [m,n], B.shape = [m,1])`

Single instance:Multiple features:Multiple Nodes:Multiple layers -> Compute $z^{[k]} = W^{[k]}x^{[k]}+B^{[k]}$ and $a^{[k]} = g^{[k]}(z^{[k]})$.

Multiple instances:Multiple features:Multiple Nodes:Multiple layers -> Compute $Z^{[k]} = W^{[k]} X^{[k]}+B^{[k]}$ and $A^{[k]} = g^{[k]}(Z^{[k]})$

> In general, the number of neurons in the previous layer gives us the number of columns of the weight matrix, and the number of neurons in the current layer gives us the number of rows in the weight matrix.

#### Activation Function

- TanH
  - Has codomain of (-1,1), is most often superior to sigmoid

- SoftMax
    > In the two-class logistic regression, the predicted probabilities are as follows, using the sigmoid function:
    >
    > 
    > $$
    > \begin{align}
    > \Pr(Y_i=0) &= \frac{e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} \, \\
    > \Pr(Y_i=1) &= 1 - \Pr(Y_i=0) = \frac{1} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}}
    > \end{align}
    > $$
    > 
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
  - $\delta a^{[2]} = \frac\delta{\delta a^{[2]}}\mathcal L(a^{[2]},y) = -\frac y {a^{[2]}} + \frac{1-y}{1-a^{[2]}}$ ($\delta\ a$ is often omittied in practice)
  - $\delta z^{[2]} = \frac{\delta}{\delta z^{[2]}}\mathcal{L}= \frac{\delta\mathcal L}{\delta a^{[2]}}\cdot \frac{\delta a^{[2]}}{\delta z^{[2]}}= \frac{\delta\mathcal L}{\delta a^{[2]}}\cdot g^{[2]\prime} = a^{[2]} - y$
  - $\delta W^{[2]} = \delta z^{[2]} \cdot a^{[1]T}$
  - $\delta b^{[2]} = \delta z^{[2]}$
  - $\delta z^{[1]}=W^{[2] T} \delta z^{[2]} \cdot g^{[1] \prime}\left(z^{[1]}\right)$
  - $\delta W^{[1]}=\delta z^{[1]} x^{T}$
  - $\delta b^{[1]}=\delta z^{[1]}$
- If the same network but has multiple training instances
  - $\delta Z^{[2]} = A^{[2]} - Y$
  - $\delta W^{[2]} = \frac 1m \delta Z^{[2]} \cdot A^{[1]T}$
  - $\delta B^{[2]} = \frac 1m np.sum( \delta z^{[2]}, axis = 1, keepdims = True)$
  - $\delta Z^{[1]}=W^{[2] T} \delta Z^{[2]} \cdot g^{[1] \prime}\left(Z^{[1]}\right)$
  - $\delta W^{[1]}= \frac 1m  \delta Z^{[1]} X^{T}$
  - $\delta B^{[1]}=np.sum(\delta Z^{[1]}, axis = 1, kepdims = True)$
- In neural network, a variable and its backward derivative have the same dimension

### Learning

Repeat derivative update untill converge:

- $w^{[\ell]} \gets w^{[\ell]} - \alpha \cdot \delta w^{[\ell]}$
- $b^{[\ell]} \gets b^{[\ell]} - \alpha \cdot \delta b^{[\ell]}$

### Random Initialization

- Weights $w$ need to be randomly initialized to mitigate symmetry breaking.
- Bias $b$ can be initialzied as $0$'s

## Deep Neural Network Notation

- Use $L$ to denote the layers in the network;
- Use $n^{[\ell]}$ to denote the number of units in layer $\ell$;
- Use $a^{[\ell]}$ to denote the activation function in layer $l$, $a^{[\ell]} = g^{[\ell]}(z^{[\ell]}) = g^{[\ell]}(w^{[\ell]} a^{[\ell - 1]} + b^{[\ell]})$

## Hyperparameters & Parameters

- Parameters
  - $W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]},...$
- Hyperparameters
  - Learning rate $\alpha$
  - \#iterations
  - \#hidden layers
  - \#hidden units
  - Choice of activation function

---

# Hyperparameter Tuning, Regularization, Optimization

## Train/Dev/Test Sets

- Learning algorithm like gradient descent use **training data** iteratively to learn the parameters of the model.
- The goal of **dev-set** is to rank the models in term of their accuracy and helps us decide which model to proceed further with.
-  We use **test-set** as a proxy for unseen data and evaluate our model on test-set.

## Bias/Variance

High variance -> Overfitting

High Bias -> Underfitting

## Regularization (Prevent Overfitting)

- $\underset{w,b}{\arg \min} \ J(w,b)$, where $J(w,b) = \frac1 m\sum^m_{i=1}L(\hat y^{(i)}, y^{(i)}), w \in \mathbb{R}^{n_x}, b \in \mathbb{R}$
- The $L2$ regularization of $J(w,b)$ is $\frac1 m\sum^m_{i=1}L(\hat y^{(i)}, y^{(i)})+ \frac\lambda{2m}||w||^2_2$, where $\lambda$ is the regularization parameter, and $||w||^2_2 = \sum^{n_x}_{j=1}w)j^2 = w^Tw$.

The Frobenuis norm of $J(w^{[1]}, b^{[1]},...,w^{[\ell]},b^{[\ell]})$ is $\frac1 m\sum^m_{i=1}L(\hat y^{(i)}, y^{(i)})+ \frac\lambda{2m}\sum^l_{l=1}||w^{[l]}||^2_F$, where $||w^{[l]}||^2_F = \sum^{n^{[l]}}_{i=1}\sum^{n^{[l-1]}}_{j=1}(w^{[l]}_{i,j})^2$

## Mini-Batch Gradient Descent

- Set the batch size:
  - Stochestic gradient descent: b_s = 1 [non-convergent, no vectorization]
  - Batch gradient descent: b_s = m [Too long per iteration]

---

# Optimize Architecture

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

---

# CNN

- Given an $n \times n$ image, $f\times f$ filter, padding $p$ and stride $s$, we have the shape of resulting convolutional matrix as $\left[\frac{n+2 p-f}{s}+1\right\rfloor \times \quad\left[\frac{n+2 p-f}{s}+1\right\rfloor$

- Convolutions over volume $n \times n \times c$, with a $f\times f\times c$ filter, will result a flattened image (sum of element-wise production).
- One layer of CNN can contain multiple kernels, each of which results a flattened output. All the result from the current layer will be stacked up for the next layer.

## Notation For Multiple-Layer CNN

- Input: $n^{[l-1]}_{H}\times n^{[l-1]}_{W}\times n^{[l-1]}_{C}$
- $f^{[l]}$ is the size of the kernel applied on input;
- $p^{[l]}$ is the extra padding adding to the input;
- $s^{[l]}$ is the increment stride for convolutioning the input;
- $n^{[l]}_C$ is the number of kernels, each of which has shape of $f^{[l]}\times f^{[l]} \times n_C^{[l-1]}$;
- $a^{[l]}$ is the output activation (without bias) which has shape of ${n}^{[l]}_{H}\times n^{[l]}_{W}\times n^{[l]}_{C}$