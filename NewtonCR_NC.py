import torch
from armijo import backForwardArmijo, backwardArmijo
from CR_NPC import CR_NPC

def NewtonCR_NC(fk, gk, hk, alpha0, alpha_npc, M, gradtol, maxite, restol, inmaxite, lineMaxite, lineBetaB, lineRho, lineBetaFB):
    xk = torch.zeros_like(gk)
    gknorm = torch.norm(gk)
    cfk, cgk, chk = cubic_f(fk, gk, hk, xk, M, "012")
    with open("Results/Results_NewtonCR_NC.txt","w") as f:
        f.write(f"test_NewtonCR_NC.py output: \n")
    for k in range(maxite):
        pk, inite, dtype = CR_NPC(chk, -cgk, restol, inmaxite)
        print("dtype: ", dtype, "inner iterations: ", inite)
        if dtype == "SOL" or dtype == "MAX":
            alphak, lineite = backwardArmijo(lambda x : cubic_f(fk, gk, hk, x, M, "0"), 
                                                        xk, cfk, cgk, alpha0, pk, 
                                                        lineBetaB, lineRho, lineMaxite)
        else:
            alphak, lineite = backForwardArmijo(lambda x : cubic_f(fk, gk, hk, x, M, "0"), 
                                                            xk, cfk, cgk, alpha_npc, pk, 
                                                            lineBetaFB, lineRho, lineMaxite)
            alpha_npc = alphak
        if torch.norm(cgk)/gknorm < gradtol:
            return xk, cfk, cgk, chk, alphak, k, inite, dtype
        xk += alphak * pk
        cfk, cgk, chk = cubic_f(fk, gk, hk, xk, M, "012")
        with open("Results/Results_NewtonCR_NC.txt","a") as f:
            f.write(f"Dtype: {dtype}, Function value: {cfk}, Gradient norm (relative): {torch.norm(cgk)/gknorm}, Outer iter: {k}, Inner iter: {inite}, Step size: {alphak}\n")

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
        if norm_y == 0:
            ch = lambda v : Av(hk, v)
        else:
            ch = lambda v : Av(hk, v) + M * (torch.inner(y, v) * y / norm_y + norm_y * v) / 2
        return cf, cg, ch

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
    hk = lambda v : random_matrix_functional(10, n, v, 1e-6, "indef", seed=43)
    M = 0.1

    alpha0 = 1
    alpha_npc = 1
    gradtol = 1e-6
    maxite = 100
    restol = 0.1
    inmaxite = 1000
    lineMaxite = 100
    lineBetaB = 1e-4
    lineRho = 0.5
    lineBetaFB = 1e-4

    NewtonCR_NC(0.0, gk, hk, alpha0, alpha_npc, M, gradtol, maxite, restol, inmaxite, lineMaxite, lineBetaB, lineRho, lineBetaFB)