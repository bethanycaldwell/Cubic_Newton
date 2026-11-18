import torch

def random_matrix_functional(n, d, v, diag, mattype, seed):
    """
    Generate a random matrix functional based on a linear combination of random vectors.
    ( H + diag I )(v)
    """
    torch.manual_seed(seed)
    if mattype == "indef":
        linear_comb = torch.randn(n, dtype=torch.float64) * 0.1
    elif mattype == "posdef":
        linear_comb = torch.rand(n, dtype=torch.float64) * 0.1
    else:
        raise ValueError("Invalid matrix type. Use 'indef' or 'posdef'.")
    w = torch.zeros(d, dtype=torch.float64)
    for i in range(n):
        a = torch.randn(d, dtype=torch.float64) * 0.1
        w += linear_comb[i] * torch.inner(a, v) * a
    w += diag * v
    return w

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
        ch = lambda v : Av(hk, v) + M * (torch.inner(y, v) * y / (2 * norm_y) + norm_y * v / 2)
        return cf, cg, ch

def Av(A , v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)

if __name__ == "__main__":
    from derivativeTest import derivativeTest
    n = 50
    gk = torch.randn(n, dtype = torch.float64)
    hk = lambda v : random_matrix_functional(n, n, v, 0.0, "posdef", seed=42)
    cubcic_f_handle = lambda y: cubic_f(0.0, gk, hk, y, 1e3, order = "012")
    derivativeTest(cubcic_f_handle, torch.ones(50, dtype = torch.float64))