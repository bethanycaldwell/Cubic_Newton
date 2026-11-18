import torch
import time
from GD import GDSolvesCubic
from NewtonCR_NC import NewtonCR_NC
from GLRT import glrt, cubic_f
from cubic_func import random_matrix_functional

def Av(A , v):
    if callable(A):
        return A(v)
    return torch.mv(A, v)

# Problem parameters
n = 100000
fk = 0.0
gk = torch.randn(n, dtype = torch.float64) * 1e-3
hk = lambda v : random_matrix_functional(10, n, v, 1e-6, "indef", seed=43)
M = 0.1

start = time.time()
GDSolvesCubic(fk, gk, hk, M, eps = 1e-6, TMax = 2000)
end = time.time()
with open("Results/Results_GD.txt","a") as f:
    f.write(f"Time taken for GD: {end - start}\n")
print("Time taken for GD: ", end - start)

# NewtonCR_NC parameters
alpha0 = 1
alpha_npc = 1
gradtol = 1e-6
maxite = 1000
restol = 1e-2
inmaxite = 1000
lineMaxite = 100
lineBetaB = 1e-4
lineRho = 0.5
lineBetaFB = 1e-4

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
hvk = Av(hk, v)
r = gk.numpy()
v = v.numpy()
hvk = hvk.numpy()
i = 1
res = {}
M = M/2

start = time.time()

with open("Results/Results_GLRT.txt","w") as f:
    f.write(f"test_GLRT.py output: \n")

while status > 0:
    status, x, r, v = glrt.solve_problem(status, n, power, M, r, v)
    inform = glrt.information()
    #res[i] = {"inform": inform, "x": x}
    with open("Results/Results_GLRT.txt","a") as f:
        f.write(str(inform)+"\n")
    i += 1
    if status == 2: # precondition v -> v/2
        for i in range(n):
            v[i] = v[i] / 2.0;
    elif status == 3: # form v -> H v
        hvk[0] = 2.0 * v[0] + v[1]
        for i in range(1,n-1):
            hvk[i] = v[i-1] + 2.0 * v[i] + v[i+1]
        hvk[n-1] = v[n-2] + 2.0 * v[n-1]
        for i in range(n):
            v[i] = hvk[i]
    elif status == 4: # reset r
        r = gk.numpy()
end = time.time()

with open("Results/Results_GLRT.txt","a") as f:
    f.write(f"Time taken: {end - start}\n")
print("Gradient norm: ",torch.norm(torch.tensor(r))/torch.norm(gk))
print("Time taken for GLRT: ", end - start)

cf = fk + torch.dot(torch.tensor(gk), torch.tensor(x)) + torch.dot(torch.tensor(hvk), torch.tensor(x)) / 2 + M * (torch.norm(torch.tensor(x)) ** 3) / 3
print("Function value: ", cf)


