import numpy as np
import math
import matplotlib.pyplot as plt

def trape(h, a, b, f):
    n = 1+int((b-a)/h)
    pts = np.linspace(a, b, n)
    sum = 0
    sum += 0.5*f(pts[0]) 
    sum += 0.5*f(pts[-1])

    for i in range(1, len(pts)-1):
        sum += f(pts[i])
    
    sum *= h

    return sum


def f1(x):
    return math.exp(x)

def f2(x):
    return x*x*(x-1)*(x-1)*math.exp(x)

if __name__ == "__main__":
    # p2b
    # ans = 1.7182818284590452353602874713526624977572470936999595749669676277
    # for h in [0.1, 0.01, 0.001, 0.0001, 1e-5]:
    #     res = trape(h, 0, 1, f1)
    #     err =  res- ans
    #     print("result is {} when h is {} as error is {}.".format(res, h, err))



    # p2c
    # ans = 14*math.e-38
    # results = []
    # errs = []
    # for h in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    #     res = trape(h, 0, 1, f2)
    #     err =  np.abs(res- ans)
    #     results.append(res)
    #     errs.append(err)
    #     print("result is {} when h is {} as error is {}.".format(res, h, err))

    # x = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
    # y = np.array(np.log(errs)) 
    # print(x, y)
    # z = np.polyfit(x, y, 2)
    # p = np.poly1d(z)
    # fit_vals = p(x)

    # plt.plot(x, y, '.-k', markersize=10, label="Raw Data Log")
    # plt.plot(x, fit_vals, 'r', label="Quadratic Fit")

    # plt.legend()
    # plt.xlabel("h")
    # plt.ylabel("log(err)")
    # plt.savefig("2b.jpg")
