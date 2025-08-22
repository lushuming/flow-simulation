from ngsolve import *
from netgen.geom2d import unit_square # mesh几何
from ngsolve.webgui import Draw
import numpy as np
from math import pi

def solve_stokes_dg(f_vec, exact_u, exact_p, epsilon=-1, sigma=10, order=1, mh=0.1):
    # 1. Construct the mesh
    mesh = Mesh(unit_square.GenerateMesh(maxh=mh))

    # 2. Define the DG space 
    X = VectorL2(mesh, order=order,dgjumps=True, dirichlet=".*")
    M = L2(mesh, order=order-1, dgjumps=True)
    V = X*M
    (u,p), (v,q) = V.TnT()

    # 3. Define the jumps and the averages
    jump_u = u - u.Other()
    jump_v = v - v.Other()
    n = specialcf.normal(2)
    mean_dudn = 0.5*(grad(u)+grad(u.Other()))*n
    mean_dvdn = 0.5*(grad(v)+grad(v.Other()))*n
    mean_p = 0.5*(p + p.Other())
    mean_q = 0.5*(q + q.Other())
    h = specialcf.mesh_size     

    # 4. Define bilinear form
    Ah = BilinearForm(V)
    Ah += InnerProduct(Grad(u), Grad(v))*dx
    Ah += (epsilon * mean_dvdn * jump_u- mean_dudn * jump_v)*dx(skeleton=True) \
        + sigma/h* jump_u*jump_v*dx(skeleton=True) 
    Ah += (epsilon * grad(v).Trace()*n*u - grad(u).Trace()*n*v)*ds(skeleton=True)  \
        + sigma/h*u*v*ds(skeleton=True)
    Ah += - p*div(v)*dx + mean_p*jump_v*n*dx(skeleton=True) + p*v*n*ds(skeleton=True)
    Ah += - q*div(u)*dx + mean_q*jump_u*n*dx(skeleton=True) + q*u*n*ds(skeleton=True)
    Ah.Assemble()
    
    # Define the right hand side
    Fh = LinearForm(V)
    Fh += InnerProduct(f_vec,v)*dx
    Fh.Assemble()
        
    # 5. Solve for the free dofs
    gf = GridFunction(V)
    gf.vec.data = Ah.mat.Inverse()*Fh.vec
    gfu, gfp = gf.components
    p_avg = Integrate(gfp, mesh) / Integrate(1, mesh)
    gfp -= p_avg  # 使压力平均值为0

    error_u = sqrt(Integrate((gfu - exact_u)**2, mesh))
    error_p = sqrt(Integrate((gfp - exact_p)**2, mesh))

    return error_u, error_p, gf.space.ndof

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'u_L2 Error':>12} | {'Order':>6} | {'p_L2 Error':>12} | {'Order':>6}")
    print("-" * 60)
    for i, (h, dofs, erroru,errorp) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {erroru:12.4e} | {'-':>6}| {errorp:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_erroru,prev_errorp = results[i-1]
            rate_u = (np.log(prev_erroru) - np.log(erroru)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_errorp) - np.log(errorp)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {erroru:12.4e} | {rate_u:6.2f}| {errorp:12.4e} | {rate_p:6.2f}")



# Define important parameters
epsilon = -1
sigma = 10 # penalty parameter
order = 2  # k
mu = 1
f_vec = CoefficientFunction((1+2*pi**3*sin(2*pi*y)*(1-2*cos(2*pi*x)), 2*pi**3*sin(2*pi*x)*(2*cos(2*pi*y)-1)))
exact_p = x-1/2
exact_u = CoefficientFunction((pi*sin(pi*x)**2*sin(2*pi*y), -pi*sin(pi*y)**2*sin(2*pi*x)))
results = []

for k in range(1, 5):
    mh = 1/2**k
    error_u, error_p, ndof = solve_stokes_dg(f_vec, exact_u, exact_p, epsilon, sigma, order, mh)
    results.append((mh,ndof,error_u, error_p ))

print_convergence_table(results)
