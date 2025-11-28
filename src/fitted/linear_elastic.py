from ngsolve import *
from netgen.geom2d import unit_square # mesh几何
from ngsolve.webgui import Draw
import numpy as np


def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_linear_elastic_fem(force, exact_u,order=1,mh=0.1):

    # 1. Construct the mesh
    mesh = Mesh(unit_square.GenerateMesh(maxh=mh))

    # 2. Define the EG space 
    fes = VectorH1(mesh, order=order, dirichlet=".*")
    u,v = fes.TnT()
    gfu = GridFunction(fes)
    
    a = BilinearForm(InnerProduct(Stress(Sym(Grad(u))), Sym(Grad(v))).Compile()*dx) 
    a.Assemble()
    f = LinearForm(force*v*dx).Assemble()
    gfu.Set(exact_u,BND)
    r = f.vec - a.mat*gfu.vec
    gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs())*r
    exact_proj = GridFunction(fes)
    exact_proj.Set(exact_u)
    L2err = sqrt(Integrate((gfu - exact_proj)**2, mesh))
    H1err = sqrt(Integrate((gfu - exact_proj)**2+(gfu.Diff(x) - exact_proj.Diff(x))**2+(gfu.Diff(y) - exact_proj.Diff(y))**2, mesh))

    
    return L2err,H1err,gfu.space.ndof

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'L2 Error':>12} | {'Order':>6}")
    print("-" * 45)
    for i, (h, dofs, error) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {error:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_error = results[i-1]
            rate = (np.log(prev_error) - np.log(error)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {error:12.4e} | {rate:6.2f}")



# Define important parameters

E, nu = 100, 0.49999
mu  = E / 2 / (1+nu)
lam = E * nu / ((1+nu)*(1-2*nu))

# 定义解析解
u_x = x*x * (1-x)*(1-x) * y*(1-y)
u_y = - x*(1-x) * y*y * (1-y)*(1-y)

# 应变分量
epsilon_xx = u_x.Diff(x)
epsilon_yy = u_y.Diff(y) 
epsilon_xy = 0.5*(u_x.Diff(y) +  u_y.Diff(x))

# 应力分量
sigma_xx = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_xx
sigma_yy = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_yy
sigma_xy = 2*mu*epsilon_xy

# 右端项 f_x, f_y
f_x = - (sigma_xx.Diff(x) + sigma_xy.Diff(y))
f_y = - (sigma_xy.Diff(x) + sigma_yy.Diff(y))

# 向量形式
exact_u = CF((u_x,u_y))
force = CF( (f_x,f_y) )

results = []
order = 3
for k in range(2, 7):
    mh = 1/2**k
    L2err,H1err,ndof = solve_linear_elastic_fem(force, exact_u,order,mh)
    results.append((mh,ndof,H1err))


print_convergence_table(results)
