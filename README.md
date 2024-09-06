# Automatic Differentiation

Automatic differentiation is a set of techniques for evaluating the partial derivatives of a function specified by a computer program. It is not symbolic differentiation.

## How it works

Automatic differentiation works by breaking down the computation of derivatives into a series of elementary operations, using the chain rule of calculus to propagate derivatives through the computation graph.

### Accumulation
The are two types of accumulation, forward and backward

In forward: $\frac{\partial w_i}{\partial x} = \frac{\partial w_i}{\partial w_{i-1}} \cdot \frac{\partial w_{i-1}}{\partial x}$

In backward: $\frac{\partial w_i}{\partial x} = \frac{\partial w_i}{\partial w_{i-1}} \cdot \frac{\partial w_{i-1}}{\partial x}$
