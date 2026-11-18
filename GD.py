import torch
from armijo import backwardArmijo

def GDSolvesCubic(fk, gk, hk, M, eps = 1e-3, TMax = 10000):
    """
    Solve the cubic regularized subproblem using gradient descent.
    
    Parameters:
    M (float): The cubic regularization parameter.
    eps (float): The tolerance for the gradient norm.
    TMax (int): The maximum number of iterations.
    
    Returns:
    A tuple (yk, cfk, i, total_oracle, cubicOpt) where:
    - yk: The solution to the subproblem.
    - cfk: The function value at yk.
    - i: The number of iterations.
    - total_oracle: The total number of oracle calls.
    - cubicOpt: The optimal value of the cubic subproblem.
    """        
    # initialization 
    gknorm = torch.norm(gk)
    gknorm2 = gknorm ** 2
    gHg = torch.dot(gk, Av(hk, gk))
    gamma = - gHg / (M * gknorm2) + torch.sqrt((gHg / (M * gknorm2)) ** 2 + 2 * torch.sqrt(gknorm2) / (M))
    yk = - gk / torch.norm(gk) * gamma
    
    cfk, cgk = cubic_f(fk, gk, hk, yk, M, order = "01") # 2 oracle calls
    total_oracle = 1
    eta = 0.5 # can change later
    with open("Results/Results_GD.txt","w") as f:
        f.write(f"test_GD.py output: \n")
    print(torch.norm(cgk, torch.inf))
    if torch.norm(cgk, torch.inf) < eps:
        return yk, cfk, 1, 2 * total_oracle, torch.norm(cgk, torch.inf)
    for i in range(TMax):
        # 2 * ite number of oracle calls
        eta, ite = backwardArmijo(lambda x : cubic_f(fk, gk, hk, x, M, order = "0"), yk, cfk, cgk, eta, -cgk, 1e-4, 0.5, 100)
        total_oracle += ite
        yk = yk - eta * cgk
        eta *= 2
        cfk, cgk = cubic_f(fk, gk, hk, yk, M, order = "01")
        print("Function value: ", cfk, "Gradient norm (relative): ", torch.norm(cgk)/gknorm, "Iteration: ", i, "Step size: ", eta)
        with open("Results/Results_GD.txt","a") as f:
            f.write(f"Function value: {cfk}, Gradient norm (relative): {torch.norm(cgk)/gknorm}, Iteration: {i}, Step size: {eta}\n")
        if torch.norm(cgk) / gknorm  < eps:
            return yk, cfk, i + 2, 2 * total_oracle, torch.norm(cgk) / gknorm
    return yk, cfk, i + 2, 2 * total_oracle, torch.norm(cgk) / gknorm

def cubic_f(fk, gk, hk, y, M, order = "01"):

    """
    Compute the cubic regularized function value and gradient at point y.
    
    Parameters:
    y (torch.Tensor): The point at which to evaluate the function.
    M (float): The cubic regularization parameter.
    order (str): The order of derivatives to compute ("0" for function value, "01" for function value and gradient).
    
    Returns:
    If order == "0": returns the function value at y.
    If order == "01": returns a tuple (function value, gradient) at y.
    """
    norm_y = torch.norm(y)
    hk_y = Av(hk, y)
    cf = fk + torch.dot(gk, y) + torch.dot(hk_y, y) / 2 + M * (norm_y ** 3) / 6
    if "0" == order: 
        return cf
    if "01" == order:
        cg = gk + hk_y + M * norm_y * y / 2
        return cf, cg
    if "012" == order:
        cg = gk + hk_y + M * norm_y * y / 2
        ch = lambda v : Av(hk, v) + M * (torch.inner(y, v) / (2 * norm_y) + norm_y * v / 2)
        return cf, cg, ch

# def cubic_f(fk, gk, hk, y, M, order = "01"):
#     """
#     Compute the cubic regularized function value and gradient at point y.
    
#     Parameters:
#     y (torch.Tensor): The point at which to evaluate the function.
#     M (float): The cubic regularization parameter.
#     order (str): The order of derivatives to compute ("0" for function value, "01" for function value and gradient).
    
#     Returns:
#     If order == "0": returns the function value at y.
#     If order == "01": returns a tuple (function value, gradient) at y.
#     """
#     norm_y = torch.norm(y)
#     hk_y = Av(hk, y)
#     cf = fk + torch.dot(gk, y) + torch.dot(hk_y, y) / 2 + M * (norm_y ** 3) / 6
#     if "0" == order: 
#         return cf
#     if "01" == order:
#         cg = gk + hk_y + M * norm_y * y / 2
#         return cf, cg

def Av(A, v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)

if __name__ == "__main__":
    import torch
    from cubic_func import random_matrix_functional
    torch.manual_seed(42)
    n = 100000
    gk = torch.randn(n, dtype = torch.float64) * 1e-3
    hk = lambda v : random_matrix_functional(10, n, v, 1e-6, "posdef", seed=43)
    M = 5

    GDSolvesCubic(0.0, gk, hk, M, eps = 1e-6, TMax = 1000)