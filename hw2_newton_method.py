import numpy as np
import matplotlib.pyplot as plt

# Does Problem 3 part (a)


def newton(f, df, x0, tol, max_iter=50):
    """ ...
        Inputs:
            f: the function for which you want to find a root
            df: derivative of aforementioned function
            x0: initialization point
            tol: error tolerance
        Returns:
            x: the approximate root
            it: the number of iterations taken
            seq: the sequence of iterates
            errs: the sequences of errors
    """
    it = 0  # iteration num
    seq = [0]*(max_iter+1) # pre allocate
    errs = []

    seq[0] = x0
    if df(seq[0]) != 0:
        seq[1] = seq[0] - f(seq[0])/df(seq[0]) 
    else: 
        print("Error: derivative is zero")
        return seq[0], it, seq

    err = seq[0] - seq[1]
    errs.append(err)
    it += 1

    while np.abs(err) > tol and it < max_iter:
        if df(seq[it]) != 0:
            seq[it+1] = seq[it] - f(seq[it])/(df(seq[it])) 
            err = seq[it+1] - seq[it]
            errs.append(err)
            it += 1
            print("Estimation at {}th iteration is {} with error={}".format(it, seq[it], err))
        else: 
            print("Error: derivative is zero")
            return seq[it], it, seq




    return seq[it], it, seq[:it], errs


def func(x):
    return (x-1)**(3/2) + x - 1


def dfunc(x):
    return ((3/2)*(x-1)**(1/2)) + 1


if __name__ == '__main__':
    # P3 (a), plotting code is not included.
    exact = 1
    x0 = 2
    x, it, seq, errs = newton(func, dfunc, x0, 1e-12)

    print('answer: {:.16f} in {} iterations.'.format(x, it))
    print("actual val is {}".format(exact))
    # print(errs)
