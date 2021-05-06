import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from copy import deepcopy

sns.set_theme()

def mat_func(f, params, X):
    """turn scalar-valued function results to matrix
    f: model function
    params: paramters
    X
    """
    m = len(X)
    pred = np.zeros((m,1))
    for i in range(m):
        pred[i, 0] = f(params, X[i])

    return pred

def derv_sim(f, params, X, row):
    """simulate derivatives for model functions (used w. LM/GN)
    f: model function
    params: paramters
    X
    row: partial derivative to be taken
    """
    params1 = params.copy()

    eps = abs(params[row]) *  np.finfo(np.float32).eps 


    params1[row] += eps
    
    p1 = f(params, X)
    p2 = f(params1, X)
     
    return (p2-p1)/(eps)

def Jacobian(f, params, X):
    """calculate Jacobian for model functions (used w. LM/GN)
    f: model function
    params: paramters
    X
    """
    m = len(X)
    n = len(params)

    J = np.matrix(np.zeros((m,n)))   
    # print(J.shape)  

    for i in range(n):
        # print(derv_sim(f, params, X, i))
        J[:,i] = derv_sim(f, params, X, i)

    return J

def loss_func(y, pred):
    """mse loss"""
    loss = np.sum(np.square(pred-y))/(2*len(y))

    return loss

def gradient(loss_func, params, X, y, f):
    """simulate gradients for loss functions (used w. GD/Newton)
    loss_func: loss function
    f: model function
    params: paramters
    X
    y
    """
    params = params.astype(float)
    N = params.shape[0]
    gradient = []
    for i in range(N):
        eps = abs(params[i]) *  np.finfo(np.float32).eps
         
        x_tmp = 1. * params[i]
        f0 = loss_func(y, mat_func(f, params, X))
        params[i] = params[i] + eps
        f1 = loss_func(y, mat_func(f, params, X))
        gradient.append(np.asscalar(np.array([f1 - f0]))/eps)
        params[i] = x_tmp

    return np.array(gradient).reshape(params.shape)

def derv_sim_grad(grad_func, loss_func, params, X, y, f, i, j):
    """simulate second-order derivatives for loss functions (used w. Newton)
    grad_func: gradient fuction 
    loss_func: loss function
    f: model function
    params: paramters
    X
    i, j: respective indices in Hessian calculation
    """
    params1 = params.copy()

    eps = abs(params[i]) *  np.finfo(np.float32).eps 

    params1[j] += eps
    
    p1 = grad_func(loss_func, params, X, y, f)[i][0]
    p2 = grad_func(loss_func, params1, X, y, f)[i][0]
     
    return (p2-p1)/(eps)

def hessian(loss_func, params, X, y, f):
    """simulate Hessians for loss functions (used w. Newton)
    loss_func: loss function
    f: model function
    params: paramters
    X
    y
    """
    N = params.shape[0]
    hessian = np.zeros((N,N)) 
    grad0 = gradient(loss_func, params, X, y, f)
    eps = np.linalg.norm(grad0) * np.finfo(np.float32).eps 
    for i in range(N):
        for j in range(N):
            hessian[i,j] = derv_sim_grad(gradient, loss_func, params, X, y, f, i,j)

    return hessian

def gradient_descent(X,y,f,theta,lr=0.1, iters=100):
    """
    f: model function
    theta: initalized paramters
    X
    y
    lr: learning rate
    iters: num of iterations
    """
    m = len(y)
    n = len(theta)
    losses = np.zeros(iters)
    thetas = np.zeros((iters,n))


    for it in range(iters):
        pred = mat_func(f, theta, X)
        loss_it = loss_func(y, pred)
        # print(loss_it)

        # dtheta = -(1/m)*lr*(X.T.dot((pred - y)))
        dtheta = -lr*gradient(loss_func, theta, X, y, f)
        # print(dtheta)
        theta += dtheta
        thetas[it,:] = theta.T
        losses[it] = loss_it

    return theta, losses, thetas

def newton(X,y,f,theta, iters=100):
    """
    f: model function
    theta: initalized paramters
    X
    y
    iters: num of iterations
    """
    m = len(y)
    n = len(theta)
    losses = np.zeros(iters)
    thetas = np.zeros((iters,n))


    for it in range(iters):
        pred = mat_func(f, theta, X)
        loss_it = loss_func(y, pred)

        g_k = gradient(loss_func, theta, X, y, f)
        h_k = hessian(loss_func, theta, X, y, f)
        
        dtheta = -(np.linalg.inv(h_k).dot(g_k))
        # print(dtheta)
        theta += dtheta
        thetas[it,:] = theta.T
        losses[it] = loss_it

    return theta, losses, thetas

def lm_method(X,y,f,theta, u=1e3, iters=100):
    """used for both LM/GN, differentiating by tuning u
    f: model function
    theta: initalized paramters
    X
    y
    u: step size
    iters: num of iterations
    """
    m = len(y)
    n = len(theta)
    losses = np.zeros(iters)
    thetas = np.zeros((iters,n))


    for it in range(iters):
        pred = mat_func(f, theta, X)
        loss_it = loss_func(y, pred)
        
        J = Jacobian(f, theta, X)

        H = J.T*J + u*np.eye(n)
        dtheta = -H.I * J.T * (pred-y)
        # print(dtheta)
        theta += dtheta
        thetas[it,:] = theta.T
        losses[it] = loss_it

        # check if loss actually decreases
        # pred_new = mat_func(f, theta, X)
        # loss_new = loss(pred_new, y)
    return theta, losses, thetas

def exe(func, method, X, y, theta_init, iters):
    """main execution function
    func: model function
    method: optimization method
    X
    y
    theta_init: initialized parameters
    iters: num of iters
    """
    if (method=="GD"):
        theta,cost_history,theta_history = gradient_descent(X,y,func, theta_init, iters=iters)
    elif (method=="Newton"):
        theta,cost_history,theta_history = newton(X,y,func, theta_init, iters=iters)
    elif (method=="GN"):
        theta,cost_history,theta_history = lm_method(X,y,func, theta_init, u=1e-10, iters=iters)
    elif (method=="LM"):
        theta,cost_history,theta_history = lm_method(X,y,func, theta_init, iters=iters)

    print("Final Theta:", theta)
    print('{} Final cost/MSE:  {:0.3f}'.format(method, cost_history[-1]))
    # print(theta_history)

    single_plot(method, iters, cost_history)

    return theta,cost_history,theta_history


def single_plot(method, iters, cost_history):
    """create single loss plots
    method: optimization method
    iters: num of iters
    cost_history: history of losses
    """
    fig,ax = plt.subplots(figsize=(12,8))

    ax.set_ylabel('J(Theta)')
    ax.set_xlabel('Iterations')
    sns.lineplot(range(iters),cost_history, legend="full")
    _=ax.plot(range(iters),cost_history,'b-')
    ax.set_ylim(0,1000)
    plt.title("{} Method Loss".format(method))
    plt.savefig("./{}.jpg".format(method))

def exe_all(func, X, y, theta_init, iters):
    """create compiled loss plots
    func: model function
    X
    y
    theta_init: initialized parameters
    iters: num of iters
    """
    theta_0 = deepcopy(theta_init)
    theta_1 = deepcopy(theta_init)
    theta_2 = deepcopy(theta_init)
    theta_3 = deepcopy(theta_init)

    theta_gd,cost_history_gd,theta_history_gd = gradient_descent(X,y,func, theta_0, iters=iters)
    theta_newton,cost_history_newton,theta_history_newton = newton(X,y,func, theta_1, iters=iters)
    theta_gn,cost_history_gn,theta_history_gn = lm_method(X,y,func, theta_2, u=np.finfo(np.float32).eps, iters=iters)
    theta_lm,cost_history_lm,theta_history_lm = lm_method(X,y,func, theta_3, iters=iters)

    fig,ax = plt.subplots(figsize=(12,8))

    ax.set_ylabel('Loss Function Value')
    ax.set_xlabel('Iterations')
    sns.lineplot(range(iters),cost_history_gd, legend='brief', label="GD")
    sns.lineplot(range(iters),cost_history_newton, legend='brief', label="Newton")
    sns.lineplot(range(iters),cost_history_gn, legend='brief', label="GN")
    sns.lineplot(range(iters),cost_history_lm, legend='brief', label="LM") 

    # _=ax.plot(range(iters),cost_history,'b-')
    # ax.set_ylim(0,1000)
    # plt.title("{} Fitting Loss".format(func.__name__))
    # plt.savefig("./{}.jpg".format(func.__name__))
    plt.title("{} Fitting Loss - Bad Init".format(func.__name__))
    plt.savefig("./{}_bad_init.jpg".format(func.__name__))



def calc_time(func, method, X, y, theta_org, iters):
    """records time
    method: optimization method
    func: model function
    X
    y
    theta_init: initialized parameters
    iters: num of iters
    """
    t = 0
    for _ in range(5): 
        tic = time.perf_counter()
        exe(func, method, X, y, theta_org, iters)
        toc = time.perf_counter()
        t += (toc-tic)
    t /= 5 
    t /= iters
    print("Execution time for method {} with {} iterations is {}".format(method, iters, t))

    return t



def linear_func(params, x):
    theta0 = params[0]
    theta1 = params[1]
    ret = theta0*x[0] + theta1*x[1]

    return ret[0]

def poly_func(params, x):
    theta0 = params[0]
    theta1 = params[1]
    theta2 = params[2]
    ret = theta0*x[0] + theta1*x[1] + theta2*x[2] 

    return ret[0]

def sin_func(params, x):
    theta0 = params[0]
    theta1 = params[1]
    ret = theta0*x[0] + theta1*np.sin(x[1]) 

    return ret[0]




if __name__ == "__main__":
    np.random.seed(0) # fix randomness
    methods = ["GD", "Newton", "GN", "LM"]
    # methods = ["GN"]
    iters = [100, 500, 1000]

    # linear func
    theta_lin = np.random.randn(2,1)
    # theta_lin = np.array([30, 6], dtype=float).reshape((2,1)) #gt
    X_lin = 5*np.random.rand(100,1)
    y_lin = 30 + 6 * X_lin + 3* np.random.randn(100,1)
    X_lin_fit = np.c_[np.ones((len(X_lin),1)),X_lin]

    # poly func
    theta_poly = np.random.randn(3,1)
    # theta_poly = np.array([10, 9, 2], dtype=float).reshape((3,1)) #gt
    # print(theta_poly)
    X_poly = np.random.rand(100,1)
    X_2_poly = X_poly**2
    y_poly = 10 + 9*X_poly + 2*X_2_poly + 1* np.random.randn(100,1)
    X_poly_fit = np.c_[np.ones((len(X_poly),1)),np.c_[X_poly, X_2_poly]]
    # print(X_poly_fit)


    theta_sin = np.random.randn(2,1)
    # theta_sin = np.array([5, 2], dtype=float).reshape((2,1)) #gt
    X = 1.5*np.random.rand(100,1)
    X_sin = np.sin(X)
    y_sin = 5 + 2*X_sin + np.random.randn(100,1)
    X_sin_fit = np.c_[np.ones((len(X_sin),1)),X_sin]
    # print(X_sin_fit)


    # 1. Time performance
    for m in methods:
        for i in iters:
            calc_time(linear_func,  m, X_lin_fit, y, theta_lin, i)
            calc_time(poly_func,  m, X_poly_fit, y_poly, theta_poly, i)
            calc_time(sin_func,  m, X_sin_fit, y_sin, theta_sin, i)


    # 2. convergence Speed 
    exe_all(linear_func, X_lin_fit, y_lin, theta_lin, 100)
    exe_all(poly_func, X_poly_fit, y_poly, theta_poly, 100)
    exe_all(sin_func, X_sin_fit, y_sin, theta_sin, 100)


    # 3. Accuracy
    exe(linear_func, "LM", X_lin_fit, y_lin, theta_lin, 1000)
    exe(poly_func, "LM", X_poly_fit, y_poly, theta_poly, 100)
    exe(sin_func, "LM", X_sin_fit, y_sin, theta_sin, 100)

    # 4. Initialization points
    theta_bad_lin = np.array([1322, -445], dtype=float).reshape((2,1))
    theta_bad_poly = np.array([13410, -39, -322], dtype=float).reshape((3,1)) 
    theta_bad_sin = np.array([1e-3,1e-3], dtype=float).reshape((2,1)) 

    exe_all(linear_func, X_lin_fit, y_lin, theta_bad_lin, 100)
    exe_all(poly_func, X_poly_fit, y_poly, theta_bad_poly, 100)
    exe_all(sin_func, X_sin_fit, y_sin, theta_bad_sin, 100)



















