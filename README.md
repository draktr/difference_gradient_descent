# FinDi: Finite Difference Gradient Descent

FinDi: Finite Difference Gradient Descent can optimize any function, including the ones without analytic form, by employing finite difference numerical differentiation within a gradient descent algorithm.

* Free software: MIT license
* Documentation: <https://findi.readthedocs.io/en/latest/>

## Installation

Preferred method to install `findi` is through Python's package installer pip. To install `findi`, run this command in your terminal

```shell
pip install findi
```

Alternatively, you can install the package directly from GitHub:

```shell
git clone -b development https://github.com/draktr/findi.git
cd findi
python setup.py install
```

## Finite Difference Gradient Descent - A Short Introduction

Finite Difference Gradient Descent (FDGD) is a modification of the regular GD algorithm that approximates the gradient of the objective function with finite difference derivatives, as

$$
-\nabla f(v) = \frac{\partial f}{\partial X} =
\begin{bmatrix}
    \frac{\partial f}{\partial x_1} \\
    \frac{\partial f}{\partial x_2} \\
    \vdots                          \\
    \frac{\partial f}{\partial x_n} \\
\end{bmatrix}
\approx
\begin{bmatrix}
    \frac{\Delta f}{\Delta x_1} \\
    \frac{\Delta f}{\Delta x_2} \\
    \vdots                          \\
    \frac{\Delta f}{\Delta x_n} \\
\end{bmatrix}
$$

Analogously, the FDGD update rule is given as

$$
v_{t+1} = v_{t} - \gamma
\begin{bmatrix}
    \frac{\Delta f}{\Delta x_1} \\
    \frac{\Delta f}{\Delta x_2} \\
    \vdots                          \\
    \frac{\Delta f}{\Delta x_n} \\
\end{bmatrix}
$$

where $\gamma$ is the same as in the regular GD. Given appropriate $\gamma$, FDGD still constructs a monotonic sequence $f(v_{0}) \geq f(v_{1}) \geq f(v_{2}) \geq \cdot \cdot \cdot$, however, due to the gradient approximation the convergence has an error proportional to the error discussed in *Differentiation* subsection. For more details refer to the Mathematical Guide in the documentation.
