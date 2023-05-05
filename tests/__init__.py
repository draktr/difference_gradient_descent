"""
FinDi

FinDi: Finite Difference Gradient Descent can optimize any function, including the ones without analytic form, by employing finite difference numerical differentiation within a gradient descent algorithm.
"""

from findi.findi import descent, partial_descent, partially_partial_descent

__all__ = [s for s in dir() if not s.startswith("_")]

__version__ = "0.1.0"
__author__ = "draktr"
