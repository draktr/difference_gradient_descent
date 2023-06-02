Examples
========

Example 1: A Simple Example
---------------

Below is a simple demonstrative example to show how to use ``findi``.
More examples can be found in `the
documentation <https://findi.readthedocs.io/en/latest/>`__, including
the examples of problems that can **only** be solved by ``findi`` and
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

Example 2: Non-Analytical, Non-Mathematical Objective Function
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
