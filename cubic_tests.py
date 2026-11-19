import torch
import time
from GD import GDSolvesCubic
from NewtonCR_NC import NewtonCR_NC
from GLRT import glrt, cubic_f
from cubic_func import random_matrix_functional, cubic_f

def Av(A , v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)

# Problem parameters
n = 1000000
fk = 0.0
gk = torch.randn(n, dtype = torch.float64) * 0.01
hk = lambda v : random_matrix_functional(10, n, v, 1e-6, "indef", seed=2026)
M = 0.1

# start = time.time()
# GDSolvesCubic(fk, gk, hk, M, eps = 1e-6, TMax = 100)
# end = time.time()
# with open("Results/Results_GD.txt","a") as f:
#     f.write(f"Time taken for GD: {end - start}\n")
# print("Time taken for GD: ", end - start)

# NewtonCR_NC parameters
alpha0 = 1
alpha_npc = 1
gradtol = 1e-6
maxite = 1000
restol = 1e-6
inmaxite = 1000
lineMaxite = 100
lineBetaB = 1e-9
lineRho = 0.5
lineBetaFB = 1e-9

start = time.time()
NewtonCR_NC(fk, gk, hk, alpha0, alpha_npc, M, gradtol, maxite, restol, inmaxite, lineMaxite, lineBetaB, lineRho, lineBetaFB)
end = time.time()
with open("Results/Results_NewtonCR_NC.txt","a") as f:
    f.write(f"Time taken for NewtonCR_NC: {end - start}\n")
print("Time taken for NewtonCR_NC: ", end - start)

# GLRT
power = 3.0
v = torch.zeros(n, dtype = torch.float64)

# set default options
options = glrt.initialize()

# set some non-default options
options['print_level'] = 2
#print("options:", options)

# load non-default options
glrt.load_options(options)

# Problem parameters for GLRT
status = 1
#hvk = Av(hk, v)
r = gk.numpy()
v = v.numpy()
#hvk = hvk.numpy()
i = 1
res = {}
M = M/2

start = time.time()

with open("Results/Results_GLRT.txt","w") as f:
    f.write(f"test_GLRT.py output: \n")

while status > 0:
    status, x, r, v = glrt.solve_problem(status, n, power, M, r, v)
    inform = glrt.information()
    with open("Results/Results_GLRT.txt","a") as f:
        f.write(str(inform)+"\n")
    i += 1
    if status == 3: # form v -> H v
        v = Av(hk, torch.tensor(v)).numpy() 
    elif status == 4: # reset r
        r = gk.numpy()
end = time.time()
print("GLRT iterations: ", i)

with open("Results/Results_GLRT.txt","a") as f:
    f.write(f"Time taken: {end - start}\n")
print("Time taken for GLRT: ", end - start)

cfk, cgk = cubic_f(fk, gk, hk, torch.tensor(x), M, order = "01")
print("GLRT", "Function value: ", cfk, "|g|: ", torch.norm(cgk, torch.inf))

#cf = fk + torch.dot(torch.tensor(gk), torch.tensor(x)) + torch.dot(Av(hk, torch.tensor(x)), torch.tensor(x)) / 2 + M * (torch.norm(torch.tensor(x)) ** 3) / 3
#print("Function value: ", cf)


