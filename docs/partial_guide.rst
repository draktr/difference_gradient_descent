
Partial Gradient Descent Guide
==============================

Partial Gradient Descent minimizes the computational expense of solving
a particular problem by reducing the number of objective function
evaluations at every epoch.

In every epoch, ``findi.partial_descent()`` uses discrete uniform
distribution to sample which parameters are going to be differentiated.
The number of parameters chosen is specified with ``parameters_used``
argument. Other parameters will be included in the update step with a
value of :math:`0`, i.e. the algorithm won’t descent (or move) in that
dimension. For example, if we have 5-dimensional parameter space and
``parameters_used=3``, then the update step in some epoch might look
like

.. math::


   v_{t+1} = v_{t} - \gamma
   \begin{bmatrix}
       \frac{0}{\Delta x_1} \\
       \frac{\Delta f}{\Delta x_2} \\
       \frac{\Delta f}{\Delta x_3} \\
       \frac{0}{\Delta x_4} \\
       \frac{\Delta f}{\Delta x_5} \\
   \end{bmatrix},

since **only** parameters :math:`x_2`, :math:`x_3`, and :math:`x_5` were
differentiated **in that epoch**. For other two parameters,
:math:`\frac{f(x+0) - f(x)}{\Delta x_i}=\frac{0}{\Delta x_i}`.

While this reduces computation time, it increases the convergence error.
This can be completely neutralized by using
``findi.partially_partial_descent()``, which uses ``partial_descent()``
for the first ``partial_epochs`` and ``descent()`` for the remained of
epochs. This method uses partial descent for rough parameter space
exploration and more accurate complete FDGD for optimum search, thereby
combining the computational benefits of partial gradient descent with
the accuracy benefits of complete finite difference gradient descent.
