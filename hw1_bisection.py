import numpy as np
from numpy import sign


def find_zero(func, a, b, tol):
    """ Use Bisection to solve the zero of a function within an interval.
        
        Args:
            func (function): function to have its zero evaluated
            a (float): left bracket of interval
            b (float): right bracket of interval
            tol (float): error tolerance

        Return:
            c (float): final approximation of the zero
            it (int): number of iterations before convergence
            approxs (int[]): list of approximations
    """
    fa = func(a) # temp vars that store func(a), func(b), and func(c)
    fb = func(b)
    fc = 0
    approxs = []  # list that saves approximations

    if sign(fa)*sign(fb) > 0:
        raise ValueError("Error message goes here.")

    it = 0
    err = 0.5*(b-a)

    while err > tol:
        c = 0.5*(a + b)
        print("The estimate at iteration {} is {} with error {}".format(it, c, err))
        fc = func(c)
        approxs.append(c)

        if sign(fa)*sign(fc) < 0:
            b = c
            fb = fc
        elif sign(fc)*sign(fb) < 0:
            a = c
            fa = fc

        it += 1
        err = 0.5*(b-a)
        

    return c, it, approxs


# hw function to be evaluated
def func(x):
    return x**3-8

def find_iter(a, b, tol):
    """Calculate actual number of iterations required to reach convergence

    Args:
        a (float): left bracket of interval
        b (float): right bracket of interval
        tol (float): error tolerance
    """
    return np.ceil(-np.log(2*tol/(b-a)))


if __name__ == "__main__":
    c, it, approxs = find_zero(func, 1.1, 2.1, tol=1e-4) # interval w. width 1 that centers around 2
    print("The number of iterations required for the function to converge at its zero when the interval is of width 1 is {}.".format(it))   

    it_act = find_iter(0, 1, 1e-4) # actual num of iterations required to reach convergence
    print("The actual number of iterations required to reach convergence is {}".format(it_act))
