"""
Answer to part b（estimations）:
Ro_10 is 0.9206264806779237
Ro_100 is 0.9992750234884827
Ro_1000 is 1.0

Answer to part c:
As n grows larger,, ro increases. However, the speed of ro increasing becomes slower. ro will remain under 1 as the method converges.
"""


import numpy as np
import copy


def norm(x):
    """2nd norm of vector

    Args:
        x (np.array): vec to have its norm calculated
    """
    return np.linalg.norm(x) 

def gs_solve(x0,b,n,tol):
    """Solve tri-diagonal matrix with -1, 2, -1 as the three diagonals.
    Apply Gauss-Seidel 

    Args:
        x0 (np.array): init point of x
        b (np.array): b in Ax = b
        n (int): mat size
        tol (int): error tol
    Return:
        x_kp1 (np.array): final result of x
        iters_mid ([np.float]): saved midpoints
    """
    iters_mid = []
    iters_mid.append(x0[n//2])
    x_k = copy.deepcopy(x0)
    iter_num = 0

    # First iter
    x_kp1 = np.zeros(n)
    x_kp1[0] = 0.5*(b[0]+x_k[1])
    for i in range(1, n-1): 
        x_kp1[i] = 0.5*(b[i]+x_kp1[i-1]+x_k[i+1])
    x_kp1[n-1] = 0.5*(b[n-1]+x_kp1[n-2])
    iters_mid.append(x_kp1[n//2])
    iter_num += 1

    while (norm(x_kp1-x_k)>=tol*norm(x_k) and iter_num<=50):
        x_k = copy.deepcopy(x_kp1)
        # First row
        x_kp1[0] = 0.5*(b[0]+x_k[1])
        # 2nd to n-1-th row
        for i in range(1, n-1): 
            x_kp1[i] = 0.5*(b[i]+x_kp1[i-1]+x_k[i+1])
        # n-th row
        x_kp1[n-1] = 0.5*(b[n-1]+x_kp1[n-2])
        iters_mid.append(x_kp1[n//2])
        iter_num += 1


    return x_kp1, iters_mid


if __name__ == "__main__":
    # working example
    n = 4 
    b = np.array([1, 2, 3, 4], dtype=float)
    x0 = np.ones(n)
    tol = 1e-10

    x, iters_mid = gs_solve(x0, b, n, tol)
    print("Solution for working example is:", x) #[4. 7. 8. 6.]




    # pb and pc
    for n in [10, 100, 1000]:
        b = np.zeros(n)
        for i in range(n): 
            b[i] = i/(n+1)**3
        tol = 1e-10
        x0 = 0.1*np.ones(n)
        
        x, iters_mid = gs_solve(x0, b, n, tol)

        # Spectral radius
        ro = (iters_mid[-1] - iters_mid[-2]) / (iters_mid[-2] - iters_mid[-3])
        print("{} iters are taken".format(len(iters_mid)))
        print("Ro_{} is {}".format(n, ro))
