import numpy as np
import copy


def lu_factor(a):
    """ computes the LU factorization of a using Gaussian elimination;
        returns a new matrix containing L and U.
        Args:
            a: the matrix to factor
        Returns:
            u : L and U, stored compactly in the lower/upper halves
    """
    # create a copy of of the matrix first for reduction.
    # note that the way you do this depends on the type you are using!
    # u = (make a copy of a)

    u = copy.deepcopy(a)
    n = len(a)

    for j in range(n-1): # reduce column 
        for i in range(j+1, n): # reduce row
            mul = u[i][j]/u[j][j]
            u[i][j:] = u[i][j:] - mul*u[j][j:]
            u[i][j] = mul

    return u


def fwd_solve_lu(a, b):
    """ Solves Lx = b where L is stored in the lower half of a """
    n = len(b)
    x = np.zeros(n)  
    for i in range(n):
        x[i] = b[i]
        for j in range(0, i):
            x[i] -= a[i][j]*x[j]
        x[i] /= a[i][i]
    return x


def back_solve_lu(a, b):
    """ Solves Ux = b where U is stored in the upper half of a """
    n = len(b)
    x = np.zeros(n)    
    for i in range(n):
        x[n-i-1] = b[n-i-1]
        for j in range(n-i, n):
            x[n-i-1] -= a[n-i-1][j]*x[j]
        x[n-i-1] /= a[n-i-1][n-i-1]
    return x


def linsolve(a, b):
    """ Solves the linear system Ax = b using Gaussian elimination.
        Args:
            a (np.array(float)): n*n matrix
            b (np.array(float)): n*1 matrix
        Returns:
            x (np.array(float)): n*1 matrix 
    """
    # compute LU factorization
    # solve Ly = b
    # solve Ux = y
    n = len(a)
    lu_mat = lu_factor(a)
    u = copy.deepcopy(lu_mat)
    l = copy.deepcopy(lu_mat)
    for i in range(n):
        for j in range(n):
            if i>j:
                u[i][j] = 0
            elif i<j:
                l[i][j] = 0
            elif i==j:
                l[i][j] = 1
    
    print("lower triangular mat: ", l, "\n")
    print("upper triangular mat: ", u)

    y = fwd_solve_lu(l, b)
    print("y matrix is ")
    print(y)
    x = back_solve_lu(u, y)


    return x


if __name__ == '__main__':
    a1 = np.array([[6, -4, 2], [-2, 2, -1], [2, -2, 2]], dtype=float)
    b1 = np.array([1, 0, 1], dtype=float)
    x1_sol = linsolve(a1, b1)
    print("Solution for p1 is:", x1_sol)

    a2 = np.array([[1,2,3,-1], [1,1,-1,2], [-1,-3,-4,5],[3,1,1,12]], dtype=float)
    b2 = np.array([1, 2, 3, 4], dtype=float)
    x2_sol = linsolve(a2, b2)
    print("Solution for p2 is:", x2_sol)

    n3 = 20
    a3 = np.zeros((n3,n3))
    b3 = np.zeros(n3)
    for i in range(n3): # Create desired matrices
        for j in range(n3):
            if i==j:
                a3[i][j] = 2
            elif np.abs(i-j)==1:
                a3[i][j] = -1
            else:
                a3[i][j] = 0
        b3[i] = i/(n3+1)**3
    
    x3_sol = linsolve(a3, b3)
    print("Solution for p3 is:", x3_sol)
    print("Maximum value is {}".format(np.max(x3_sol)))


    a4 = np.array([[2,-1,0,0], [-1,2,-1,0], [0,-1,2,-1],[0,0,-1,2]], dtype=float)
    b4 = np.array([1, 2, 3, 4], dtype=float)
    x4_sol = linsolve(a4, b4)

    print("Solution for a4 is:", x4_sol)





