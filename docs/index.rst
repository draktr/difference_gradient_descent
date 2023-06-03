FinDi: Finite Difference Gradient Descent
=========================================

FinDi: Finite Difference Gradient Descent can optimize any function,
including the ones without analytic form, by employing finite difference
numerical differentiation within a gradient descent algorithm.

-  Free software: MIT license
-  Documentation: https://findi-descent.readthedocs.io/en/latest/

Installation
------------

A preferred method to install ``findi`` is through Python’s package
installer pip. To install ``findi``, run this command in your terminal

.. code:: shell

   pip install findi-descent

Alternatively, you can install the package directly from GitHub:

.. code:: shell

   git clone -b development https://github.com/draktr/findi-descent.git
   cd findi-descent
   python setup.py install

Finite Difference Gradient Descent - A Short Introduction
---------------------------------------------------------

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
proportional to the error discussed in *Differentiation* subsection. For
more details refer to the Mathematical Guide in the documentation.

Features
--------

Optimization Algorithms
~~~~~~~~~~~~~~~~~~~~~~~

-  ``descent()`` - regular FDGD algorithm
-  ``partial_descent()`` - FDGD algorithm where in each epoch
   ``parameters_used`` number of parameters are randomly selected to be
   differenced. This approach is useful in cases where the evaluation of
   objective function is computationally expensive
-  ``partially_partial_descent()`` - FDGD algorithm that uses
   ``partial_descent()`` algorithm for the first ``partial_epochs``
   number of epochs and ``descent()`` for the remaining epochs. This
   approach is useful as it combines the computational efficiency of
   ``partial_descent()`` with the approximational accuracy of
   ``descent()``

Computational
~~~~~~~~~~~~~

-  Numba mode - Numba just-in-time compilation is available for **all**
   algorithms, including automatically parallelized evaluation. This
   drastically decreases computation time, however, it also requires the
   objective function to be Numba-compiled
-  ``joblib`` parallelization - supported in Python mode. This is
   helpful, especially with high-dimensional problems where Numba
   objective function is unfeasible
-  ``values_out()`` function - exports outputs, parameters, and
   constants values for each epoch as ``Pandas`` ``DataFrame``. This is
   useful for, among other things, algorithm convergence analytics and
   hyperparameter (e.g. learning rates and differences) tuning
-  Variable learning rates and difference values - other than scalars,
   ``l`` and ``h`` arguments also accept array_like structures
   (e.g. lists and ``Numpy`` arrays). These can be constructed manually,
   by the library
   `OptSchedule <https://pypi.org/project/optschedule/>`__ which
   provides a variety of decay schedules
-  ``momentum`` hyperparameter - accelerates gradient descent in the
   relevant direction and dampens oscillations. ``momentum = 0``
   (default value) implies no acceleration and dampening. The update
   rule with ``momentum > 0`` is

.. math::


   v_{t} = m*v_{t-1} - l_{t} * \frac{F(X_{t})-F(X_{t-1})}{h}
   X_{t} = X_{t-1} + v_{t}

Non-Mathematical Functions as Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  support for **metaparameters** - FinDi accepts objective functions
   that require *metaparameters* to be passed to it. By *metaparameter*
   is considered any parameter passed to the objective function that
   **will not** be differenced and its value will be held constant
   throughout epochs
-  support for **multiple outputs** - FinDi accepts objective functions
   that return more than one value. For example, if the objective
   function has a convex optimization routine within it, FinDi allows
   for the objective function to return the regular objective value
   along with the solutions to the optimization problem. The first value
   of the return structure will be taken as the objective value to be
   minimized

Advantages Over Other Optimization Techniques
---------------------------------------------

1) Optimizing objective functions that **cannot be expressed or solved
   analytically** or **discontinuous functions**
2) **Intuitive and easy to communicate** its implementation, unlike most
   of the derivative-free optimization methods
3) Convenient work with blackbox or proprietary objective functions
   through metaparameters, where source code might be inaccessible
4) Increased computational efficiency with Numba **just-in-time
   compilation**
5) Supports **parallelization** via ``joblib`` or ``numba`` library
6) **Partial Gradient Descent** makes high-dimensional, simple problems
   less computationally expensive to solve
7) Built-in support for **variable learning rates and differences**

A Quick Example
---------------

Below is a simple demonstrative example to show how to use ``findi``.
More examples can be found in `the
documentation <https://findi-descent.readthedocs.io/en/latest/>`__,
including the examples of problems that can be solved by ``findi`` and
not by other Python Gradient Descent implementations.

.. code:: python

   import findi as fd

   # Defining the objective function
   def foo(params):
       return [(params[0]+2)**2]

   # Descent
   outputs, parameters = fd.descent(
       objective=foo,
       initial=5,
       h=0.0001,
       l=0.01,
       epochs=1000,
   )

   print("Solution (argmin): ", parameters[-1])
   print("Objective value at solution (min): ", outputs[-1])

   # Saves values of outputs and parameters as Pandas DataFrame...
   values = fd.values_out(outputs, parameters, columns=["x"])
   # ...to be stored as a CSV file
   values.to_csv("values.csv")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   findi

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   math_guide
   computational_features_guide
   partial_guide
   variable_rates_guide

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples

Project Principles
------------------

-  Easy to be understood and used by those with little computer science
   background, including scientists, researchers and industry
   practitioners
-  Flexibility for proprietary modifications
-  Emphasis on computational efficiency
-  Use consistency across approaches (Numba vs Python, regular Gradient
   Descent vs Partial Gradient Descent etc.)
-  Tested
-  Dedicated and detailed technical and applicative documentation
-  Formatting deferred to `Black <https://github.com/psf/black>`__

Future Development
------------------

Feature requests are more than welcome through the Issues forum, as are
bug reports and improvement recommendations. Anyone is more than welcome
to become a contributor or maintainer.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
