
Mathematical Guide
==================

Introduction
------------

This serves as technical documentation for the mathematical background
behind FinDi: Finite Difference Gradient Descent Python optimizer.
Regular gradient descent will be briefly explained first, followed by
its finite difference version, as implemented in FinDi. Lastly, an
example will be shown of a problem that cannot be solved by a regular
gradient descent optimizer.

(Regular) Gradient Descent
--------------------------

Gradient Descent (GD) is an iterative algorithm for finding a local
minimum of a function. The basis of GD is that the objective function
:math:`f` decreases the fasters in the direction of its negative
gradient at a particular point :math:`v`. Note that a gradient is
defined as

.. math::


   -\nabla f(v) = \frac{\partial f}{\partial X} =
   \begin{bmatrix}
       \frac{\partial f}{\partial x_1} \\
       \frac{\partial f}{\partial x_2} \\
       \vdots                          \\
       \frac{\partial f}{\partial x_n} \\
   \end{bmatrix}

and gradient descent update rule is given by

.. math::


   v_{t+1} = v_{t} - \gamma \nabla f(v_{t}),

where :math:`\gamma` is a learning rate and
:math:`\gamma \in \mathbb{R_{+}}.` Mathematically speaking, given
initial guess :math:`v_{0}`, GD constructs a monotonic sequence
:math:`f(v_{0}) \geq f(v_{1}) \geq f(v_{2}) \geq \cdot \cdot \cdot` that
should converge to a local minimum, given appropriate learning rate
:math:`\gamma`.

Problems of (regular) Gradient Descent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1) Objective function :math:`f` needs to be differentiable, i.e. a
   function that has a **cusp** or any type of **discontinuity**
   wouldn’t work. Moreover, **any function without analytical form** or
   without analytical gradient also is not solvable with the regular
   gradient descent.
2) Sensitivity to learning rate :math:`\gamma`
3) For GD to find the global minimum, :math:`f` needs to be **convex**.
   Otherwise, GD might converge to a local minimum (or not converge at
   all).

Finite Difference Gradient Descent
----------------------------------

Differentiation
~~~~~~~~~~~~~~~

The **Finite difference operator** maps functions to functions and is
defined as

.. math::


   \Delta_{h}[f] = f(x+h)-f(x).

The above expression is called the **forward difference** due to
positive propagation. Analogously, the **backward difference** is
defined as

.. math::


   \Delta_{h}[f] = f(x)-f(x-h),

and the **central difference** is defined as

.. math::


   \Delta_{h}[f] = f(x+\frac{h}{2})-f(x-\frac{h}{2}).

Remember the pre-analysis definition of a **derivative**:

.. math::


   \frac{\partial f}{\partial x} := \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}.

Usefully to us finite difference approximates derivatives by, instead of
using infinitesimal changes, it uses finite changes. Let :math:`h` be
some finite, non-zero value, then

.. math::


   \lim_{h \to 0} \frac{f(x+h)-f(x)}{h} \approx \frac{f(x+h)-f(x)}{h} = \frac{\Delta_{h}[f]}{h}

Since :math:`\frac{\Delta_{h}[f]}{h}=\frac{\Delta f}{\Delta x}`

.. math::


   \frac{\partial f}{\partial x} \approx \frac{\Delta f}{\Delta x}.

This works for backward and central difference as well.

The error of this approximation can be obtained with Taylor’s theorem.
By rearranging the Taylor series expansion

.. math::


   \frac{f(x+h) - f(x)}{h} - f'(x) = h \frac{f''(\eta)}{2},

where :math:`\eta \in (x, x+h)`.

While this error is the same for backward difference approximation, the
error for central difference approximation is smaller and is given by

.. math::


   \frac{f(x+\frac{h}{2})-f(x-\frac{h}{2})}{h} = h^{2} \frac{f'''(\eta_{1})+f'''(\eta_{2})}{12}

where :math:`\eta_{1} \in (x, x+h)` and :math:`\eta_{2} \in (x-h, x).`

Gradient Descent
~~~~~~~~~~~~~~~~

Finite Difference Gradient Descent (FDGD) is a modification of the
regular GD algorithm that approximates the gradient of the objective
function with finite difference derivatives, as

.. math::


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

Analogously, the FDGD update rule is given as

.. math::


   v_{t+1} = v_{t} - \gamma
   \begin{bmatrix}
       \frac{\Delta f}{\Delta x_1} \\
       \frac{\Delta f}{\Delta x_2} \\
       \vdots                          \\
       \frac{\Delta f}{\Delta x_n} \\
   \end{bmatrix}

where :math:`\gamma` is the same as in the regular GD. Given appropriate
:math:`\gamma`, FDGD still constructs a monotonic sequence
:math:`f(v_{0}) \geq f(v_{1}) \geq f(v_{2}) \geq \cdot \cdot \cdot`,
however, due to the gradient approximation the convergence has an error
proportional to the error discussed in *Differentiation* subsection.

An Example
----------

Below is an example of a problem that minimizes the sum of squares
between some data and some values that can be or are a function of
parameters. Since we cannot find a gradient of the objective function,
we cannot use any regular implementation of gradient descent. However,
FinDi can solve this problem easily as it only needs the objective to be
evaluated to optimize it.

.. code:: python

   import numpy as np
   import findi as fd
   import cvxpy

   # Defining the objective function
   def objective(params):
       data = np.array([5, 2, 7])
       values = np.vstack((np.array(params[0:3]), np.array(params[3:6]))).transpose()

       x = cvxpy.Variable(2)
       objective = cvxpy.Minimize(cvxpy.sum_squares(values @ x - data))
       constraints = [0 <= x, x <= 1, cvxpy.sum(x) == 1]
       prob = cvxpy.Problem(objective, constraints)
       result = prob.solve()

       return prob.value, x.value[0], x.value[1]

   # Descent
   outputs, parameters = fd.descent(
       objective=objective,
       initial=np.array([0, 0, 0, 0, 0, 0]),
       h=0.001,
       l=0.01,
       epochs=1000,
   )

   print("Solution (argmin): ", parameters[-1])
   print("Objective value at solution (min): ", outputs[-1])

Summary
-------

Finite difference gradient descent uses a simple modification of the
regular gradient descent to be able to optimize a much wider variety of
objective functions. Unlike derivative-free optimization methods, FDGD
is intuitive making it easier to integrate in scientific, engineering
and commercial projects.
