# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:19:31 2024

@author: uqalim8
"""
import torch

def CR_NPC(A, b, rtol, maxit):    
    x, p, r = torch.zeros_like(b), b, b
    Ap = Avec(A, p)
    Ar = Ap.clone()
    rAr = torch.dot(r, Ar)
    Apsq = torch.dot(Ap, Ap)
    
    normb = torch.norm(r)
    normrsq = normb ** 2
    normAb = torch.norm(Ar)
    normArsq = normAb ** 2

    k = 1
    if rAr < 0:
        return r / normb, k, "NPC"
    
    while torch.sqrt(normrsq) > rtol * normb and torch.sqrt(normArsq) > rtol * normAb and k < maxit:
        alpha = rAr / Apsq
        x = x + alpha * p
        rp1 = r - alpha * Ap

        Arp1 = Avec(A, rp1)
        rp1Arp1 = torch.dot(rp1, Arp1)
        normArp1sq = torch.norm(Arp1) ** 2
        beta = rp1Arp1 / rAr
        p = rp1 + beta * p
        Ap = Arp1 + beta * Ap
                
        # update
        normrsq = normrsq - alpha * rAr
        Apsq = normArp1sq - beta ** 2 * Apsq
        normArsq = normArp1sq
        Ar = Arp1
        rAr = rp1Arp1
        r = rp1

        k += 1
        if rAr < 0:
            return r / torch.sqrt(normrsq), k, "NPC"
            
    if k == maxit:
        return x, k, "MAX"
        
    return x, k, "SOL"

def Avec(A, x):
    if callable(A):
        return A(x)
    return torch.mv(A, x)

if __name__ == "__main__":
    from cubic_func import random_matrix_functional
    N = 100
    H = lambda v : random_matrix_functional(10, N, v, 1e-6, "posdef", seed=42)
    b = torch.rand(N, dtype = torch.float64)
    x, k, dtype = CR_NPC(H, b, 1e-6, N)
    print("Iterations: ", k, "dtype: ", dtype)
    print(torch.norm(Avec(H, Avec(H, x) - b)))