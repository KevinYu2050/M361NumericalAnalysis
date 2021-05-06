import numpy as np
import copy

def trilu(A):
    """ ...

        Args:
            A: the input matrix as an nx3 array of bands

        Returns:
            factors: the lu decomposition, also as an nx3 array with the first
                column containing the lower diag. of L and the other two
                columns containing the central/upper diagonals of U.
    """
    n = A.shape[0]
    factors = copy.deepcopy(A)

    for i in range(1,n): # reduce column 
        mul = factors[i][0]/factors[i-1][1]
        factors[i][1] = factors[i][1] - mul*factors[i-1][2]
        factors[i][0] = mul

    return factors

def fwd_solve_lu(lu, b):
    """ Solves Lx = b where L is stored in the lower half of a """
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0]
    for i in range(1,n):
        x[i] = b[i]
        x[i] -= lu[i][0]*x[i-1]
    return x


def back_solve_lu(lu, b):
    """ Solves Ux = b where U is stored in the upper half of a """
    n = len(b)
    x = np.zeros(n)  
    x[n-1] = b[n-1]/lu[n-1][1]
  
    for i in range(0,n-1):
        x[n-2-i] = b[n-2-i]
        x[n-2-i] -= lu[n-2-i][2]*x[n-1-i]
        x[n-2-i] /= lu[n-2-i][1]
    return x


def trisolve(A, b):
    """ ...

        Args:
            A: the input matrix as an nx3 array of bands
            b: the RHS vector (length n)

        Returns:
            x: the solution to Ax = b
    """

    factors = trilu(A)
    n = A.shape[0]

    # forward/back solves
    y = fwd_solve_lu(factors, b)
    x = back_solve_lu(factors, y)

    return x


def diff_matrix(n):
    """ construct the matrix for the example solve in banded form """
    mat = np.zeros((n, 3), dtype=float)
    for k in range(n):
        mat[k, 0] = -1  # a_(k,k-1)
        mat[k, 1] = 2  # a_(k,k)
        mat[k, 2] = -1  # a_(k,k+1)

    # REMARK: the upper left/ lower right entries are *unused*,
    # so diff[0, 0] and diff[n-1, 2] could have any value.
    # It's nice to set them to zero here for clarity:

    mat[0, 0] = 0
    mat[n-1, 2] = 0

    return mat

def test_b(n):
    """ construct the b for the example solve in banded form """
    b = np.zeros((n, 1), dtype=float)
    for i in range(n):
        b[i] = i/(n+1)**3
    
    return b


if __name__ == '__main__':
    test_b = test_b(1000)
    res = trisolve(diff_matrix(1000), test_b)
    print("The solution for the matrix is: \n")
    print(res)
    print("Maximum value is {}".format(np.max(res)))
