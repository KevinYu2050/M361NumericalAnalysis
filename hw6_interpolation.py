import numpy as np
import matplotlib.pyplot as plt
import math

def bary_weights(nodes):
    """[summary]

    Args:
        nodes ([np.array]): input nodes
    """
    n = len(nodes)
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= 1/(nodes[i] - nodes[j]) 

    return weights


def bary_polyval(nodes,w,f,xvals):
    """[summary]

    Args:
        nodes ([np.array]): input nodes
        w ([np.array]): input weights
        f ([func]): input values at each of the nodes
        xvals ([np.array]): values to be evaluated
    """
    n = len(xvals)
    numerators = np.zeros(n)
    denominators = np.zeros(n)
    ret = np.zeros(n)

    for i in range(n):
        if not np.any(nodes == xvals[i]):
            for j in range(len(w)):
                numerators[i] += w[j]*f[j]/(xvals[i]-nodes[j])
                denominators[i] += w[j]/(xvals[i]-nodes[j])
            ret[i] = numerators[i]/denominators[i]
        else:
            ind, = np.where(nodes == xvals[i])
            print(ind)
            ret[i] = f[ind[0]]
    
    return ret

def func_b(x):
    return 1/(1+x**2)

def func_c(x):
    return 1/(1+25*x**2)

def test_func(func):
    errs = []
    for n in range(5, 40, 2):
        nodes = np.linspace(-1, 1, n)
        fvals = np.ones(n)
        for i in range(n):
            fvals[i] = func(nodes[i])
        weights = bary_weights(nodes)

        xvals = np.linspace(-1, 1, n+5)
        interpolates = bary_polyval(nodes, weights, fvals, xvals)
        true_ys = np.ones(n+5)
        for i in range(n+5):
            true_ys[i] = func_b(xvals[i])
        print(interpolates)
        err = np.max(np.abs(interpolates-true_ys))
        errs.append(err)
    
    return errs

def test_func_chebyshev(func):
    errs = []
    for n in range(5, 20, 2):
        nodes = chebyshev(n)
        fvals = np.ones(n)
        for i in range(n):
            fvals[i] = func(nodes[i])
        weights = bary_weights(nodes)

        xvals = np.linspace(-1, 1, n+5)
        interpolates = bary_polyval(nodes, weights, fvals, xvals)
        true_ys = np.ones(n+5)
        for i in range(n+5):
            true_ys[i] = func_b(xvals[i])
        print(interpolates)
        err = np.max(np.abs(interpolates-true_ys))
        errs.append(err)
    
    return errs

def chebyshev(n):
    nodes = np.zeros(n)
    for i in range(n):
        nodes[i] = math.cos(math.pi*(2*i+1)/(2*n))

    return nodes


if __name__ == "__main__":
    # part b
    errs = test_func(func_c)

    # x = np.array(list(range(5, 40, 5)))
    # y = np.array(np.log(errs)) 
    # # print(y)


    # part c 
    # errs = test_func_chebyshev(func_c)

    x = np.array(list(range(5, 40, 2)))
    y = np.array(np.log(errs)) 
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(slope, intercept)

    plt.plot(x, y, '.-k', markersize=10, label="Raw Data Log")
    # plt.plot(x, slope*x+intercept, 'r', label="Linear Fit: y=-0.57x-1.466")
    # plt.plot(x, slope*x+intercept, 'r', label="Linear Fit: y=-0.012x-0.18")
    plt.plot(x, slope*x+intercept, 'r', label="Linear Fit: y=0.249x-2.45")

    plt.legend()
    plt.xlabel("n")
    plt.ylabel("log(maximum err)")
    plt.savefig("c1_c.jpg")


