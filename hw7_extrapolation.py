import numpy
import math

def richardson( f, x, n, h ):
    """Richardson's Extrapolation to approximate  f''(x) at a particular x.

    USAGE:
	d = richardson( f, x, n, h )

    INPUT:
	f	- function to find derivative of
	x	- value of x to find derivative at
	n	- number of levels of extrapolation
	h	- initial stepsize

    OUTPUT:
        numpy float array -  two-dimensional array of extrapolation values.
                             The [n,n] value of this array should be the
                             most accurate estimate of f''(x).

    NOTES:                             
        Based on an algorithm in "Numerical Mathematics and Computing"
        4th Edition, by Cheney and Kincaid, Brooks-Cole, 1999.

    AUTHOR:
        Jonathan R. Senning <jonathan.senning@gordon.edu>
        Gordon College
        February 9, 1999
        Converted ty Python August 2008
    """

    # d[n,n] will contain the most accurate approximation to f'(x).

    d = numpy.array( [[0] * (n + 1)] * (n + 1), float )
    ret = numpy.zeros(n+1)
    errs = numpy.zeros(n+1)

    for i in range( n + 1 ):
        d[i,0] =  ( f( x + h ) + f( x - h ) -2*f(x) ) / (h*h)

        powerOf4 = 1  # values of 4^j
        for j in range( 1, i + 1 ):
            powerOf4 = 4 * powerOf4
            d[i,j] = d[i,j-1] + ( d[i,j-1] - d[i-1,j-1] ) / ( powerOf4 - 1 )

        h = 0.5 * h
        ret[i] = d[i,i]
        errs[i] = 4*(d[i,i] - d[i, i//2])/3

    return ret, errs

def ex(x):
    return math.exp(2*x)

if __name__ == "__main__":
    M = 20
    with open("res.txt", "a") as f:
        for m in range(1,M+1):
            richardson_res, errs = richardson(ex, 1, M, 2**(-m))
            for lev in range(len(richardson_res)):
                line = "for m={}, d{}={}, err={}".format(m, lev, richardson_res[lev], errs[lev])
                print(line)
                f.write(line + "\n")