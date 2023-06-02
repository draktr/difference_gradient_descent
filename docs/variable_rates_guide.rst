Variable Learning Rates and Difference Values Guide
===================================================

Choosing an appropriate learning rate can be challenging, especially for
more complex optimization problems. Setting the learning rate too low
and the algorithm won’t be able to explore the objective range while
setting it too high and the algorithm will overshoot the optimum. The
solution to this is to use a decay sequence where each next epoch will
have a learning rate lower than the previous one. This way in earlier
epochs higher learning rate will allow for exploration of the objective
range, while in later epochs lower learning rate will help with
convergence.

As seen in Mathematical Guide, finite difference differentiation error
is a function of the difference value. If the difference is too large,
one will have a bad estimate of the derivative, while if the difference
value is too small there might be cancellation issues or large rounding
errors. Optimal difference value can rarely be mathematically determined
especially for problems with non-analytical objective function. By using
a decay sequence (including piecewise constant sequences), differences
are decreased to some terminal value, minimizing the aforementioned
issues.

FinDi supports both variable learning rates and difference values. Decay
sequences can be easily constructed with
`OptSchedule <https://pypi.org/project/optschedule/>`__ library and
passed to ``l`` and ``h`` arguments, respectively.
