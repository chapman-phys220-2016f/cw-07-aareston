#!/usr/bin/env python3

import numpy as np
import nose as n

"""
Solves a nonlinear ODE using Euler's forward method
"""

def euler(f = lambda t, x: 0, f_0 = 0, t_0 = 0, t_f = 1, delta_t = 0.1):
    """
    Applies Euler's forward method to a given first order differential equation of the form f' = h(t, f(t))

    Parameters:
    -----------
        f: lambda t, f_t:
            t is time parameter
            f_t is considered only if the derivative is dependent on the function itself (as with exponentials)
            differential equation solved for the derivative of the function
        f_0: float
            initial condition for the function
        t_0: float
            initial independent variable value
        t_f: float
            endpoint for applying the method
        delta_t: float
            step size for method implementation

    Returns:
    --------
        yvals: np.array
            array of yvals for numerical solution
    """
    xvals = np.arange(t_0, t_f, delta_t)
    yvals = np.zeros(xvals.size)
    yvals[0] = f_0
    for i in range(1, xvals.size):
        yvals[i] = yvals[i - 1] + f(xvals[i - 1], yvals[i - 1]) * (delta_t)
    return yvals

def test_euler():
    "Tests euler() for differential equation y' = 1, y_0 = 1, delta_t = 0.001, a = 0, b = 0.1"
    test = euler(lambda t, x: x, 1, 0, 0.1, 0.001)
    xvals = np.arange(0,0.1,0.001)
    case = np.exp(xvals)
    success = 0
    def a_eq(a,b,eps = 1E-3):
        return abs(a-b) < eps

    for (i, j) in zip(test,case):
        if a_eq(i,j):
            success += 1

    assert success == xvals.size

def sample_function_error(f, g, y_0, delta_t, a, b):
    """
    Calculates RMS error between a given function and its approximate solution

    Parameters:
    -----------
        f: lambda
            differential equation to solve
        g: lambda
            actual solution
        y_0: float
            initial condition for particular solution
        delta_t: float
            step size for approximation
        a: float
            start point for approximation
        b: float
            end point for approximation

    Returns:
    --------
        rms: float
            root mean squared error between approximation and actual solution
    """
    sample = euler(f, y_0, a, b, delta_t)
    xvals = np.arange(a, b, delta_t)
    actual_y = g(xvals)
    rms = np.sqrt(np.sum((sample - actual_y) ** 2))
    return rms

def test_sample_function_error():
    "Tests calculation of rms for case y' = 3t, y_0 = 0, delta_t = .5, a = 0, b = 1"
    test = sample_function_error(lambda t, x: 3 * t, lambda t: 3 * t ** 2 / 2.0 + 1, 1, 0.3, 0, 1)
    case = 0.505124
    n.tools.assert_almost_equals(test, case, places = 4)
