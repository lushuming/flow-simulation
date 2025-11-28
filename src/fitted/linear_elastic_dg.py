from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
from math import pi,e
from numpy import linspace
import numpy as np
import scipy.sparse as sp

ngsglobals.msg_level = 2

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_linear_elastic_dg(h0, quad_mesh, orderu,fe, exact_u, mu,lam,beta_u):

    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2. Construct the  DG space 
    U = VectorL2(mesh, order=orderu, dirichlet=[], dgjumps=True) # space for displacement
    u,v = U.TnT()

    # Define special variables
    ne = specialcf.normal(mesh.dim) # normal vectors on faces
    h = specialcf.mesh_size  

    strain_u = Sym(Grad(u))
    strain_v = Sym(Grad(v))
    mean_stress_u = 0.5*(Stress(Sym(Grad(u)))+Stress(Sym(Grad(u.Other()))))*ne
    mean_stress_v = 0.5*(Stress(Sym(Grad(v)))+Stress(Sym(Grad(v.Other()))))*ne
    jump_u = u - u.Other()
    jump_v = v - v.Other()
    # jump_du = Grad(u) - Grad(u.Other())
    # jump_dv = Grad(v) - Grad(v.Other())
    # integration domains:

    # 4. Construc bilinear form and right hand side 

    Ah = BilinearForm(U)
    # Ae
    Ah += 2*mu*InnerProduct(strain_u,strain_v)*dx + lam*div(u)*div(v)*dx \
            - (InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dx(skeleton=True) \
            - (InnerProduct(Stress(Sym(Grad(u)))*ne,v) + InnerProduct(Stress(Sym(Grad(v)))*ne,u) - beta_u/h*InnerProduct(u,v))*ds(skeleton=True)
    Ah.Assemble()

    # r.h.s
    lh = LinearForm(U) 
    lh += fe*v*dx - InnerProduct(uD,Stress(Sym(Grad(v)))*ne)*ds(skeleton=True) + beta_u/h*uD*v*ds(skeleton=True)
    lh.Assemble()

    gfu = GridFunction(U)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec

    
    # Calculate the condition number of the stiffness matrix
    rows,cols,vals = Ah.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    # conds = np.linalg.cond(A.todense())
    conds = 0
        
    # 5. Calculate L2 error
    error_u = sqrt(Integrate((gfu - exact_u)**2 * dx, mesh))

    gff = GridFunction(U)
    gff.Set(exact_u)
    grad_error_u = Grad(gfu - gff)
    error_u_H1 = sqrt(Integrate(InnerProduct(grad_error_u,grad_error_u)*  dx, mesh))

    return error_u,error_u_H1,gfu.space.ndof, conds

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'Cond.':>12} | {'Error_u_L2.':>12}  | {'Order':>6} | {'Error_u_H1.':>12}  | {'Order':>6} ")
    print("-" * 70)
    for i, (h,dofs,conds,error_u,error_u_H1) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{error_u:12.4e} | {'-':>6} | {error_u_H1:12.4e} | {'-':>6}")
        else:
            prev_h,_,_, prev_error_u, prev_error_u_H1 = results[i-1]
            rate_u = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
            rate_u_H1 = (np.log(prev_error_u_H1) - np.log(error_u_H1)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{error_u:12.4e}| {rate_u:6.2f} | {error_u_H1:12.4e}| {rate_u_H1:6.2f} ")



# Define important parameters
# physical parameters for linear elastic
mu  = 10
lam = 100

# parameters of DG method
order_u = 2
beta_u = 200

quad_mesh = False


# manufactured solution
u_x = sin(pi*x)*sin(pi*y)
u_y = x*y*(x-1)*(y-1)

exact_u = CF((u_x, u_y))

# strain tensor
epsilon_xx = u_x.Diff(x)
epsilon_yy = u_y.Diff(y) 
epsilon_xy = 0.5*(u_x.Diff(y) +  u_y.Diff(x))


# total stress tensor
sigma_xx = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_xx 
sigma_yy = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_yy 
sigma_xy = 2*mu*epsilon_xy



# 右端项 f_x, f_y
f_x = - (sigma_xx.Diff(x) + sigma_xy.Diff(y))
f_y = - (sigma_xy.Diff(x) + sigma_yy.Diff(y))

fe = CF((f_x, f_y))

uD = exact_u

# Set level set function
levelset = sqrt(x**2 + y**2) - 1/2

results = []

for k in range(2, 6):
    h0 = 1/2**k
    error_u,error_u_H1, ndof, conds = solve_linear_elastic_dg(h0, quad_mesh, order_u,fe, exact_u, mu,lam,beta_u)
    results.append((h0,ndof,conds,error_u,error_u_H1))

print_convergence_table(results)
