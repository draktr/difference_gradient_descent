
Computational Features Guide
============================

Numba Mode
----------

`Numba <https://numba.pydata.org/>`__ is an open source just-in-time
compiler that translates a subset of Python and Numpy code into machine
code, thereby significantly decreasing computation time. Numba mode is
available for **all** ``findi`` algorithms by specifying the argument
``numba=True``. In numba mode **parallelization is automatically
handled** by Numba. It should be noted, however, that since Numba
completely skips the Python interpreter when computing, it requires the
objective function to be Numba-compiled and by extension
Numba-compatible. Numerical computation functions generally are or can
be made Numba-compatible, but for more information refer to `Numba
documentation <https://numba.readthedocs.io/en/stable/>`__.

Parallelization Using ``joblib`` Library
----------------------------------------

In Python mode (default, ``numba=False``) parallelization of the
evaluation of objective functions is handled by ``joblib`` library. This
is generally useful in problems with high-dimensional parameter spaces
or where the objective function cannot be rewritten to be
Numba-compatible. Note, however, that due to overheads ``joblib``
parallelization is not beneficial for simple and computationally
inexpensive objective functions.
