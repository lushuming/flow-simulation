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

def solve_biot_steady_unfitted(h, quad_mesh, order, levelset, b, f, exact_u, exact_p, mu,lam,tau_fpl,lambda_u,lambda_p,gamma_s,gamma_p,gamma_m,alpha,M,K):

    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=h, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=1, threshold=0.1,
                                      discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1
    InterpolateToP1(levelset,lsetp1)

    # Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    # facets used for ghost penalty stabilization:
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)

    # 3. Construct the unfitted fem space 
    Uhbase = VectorH1(mesh, order=order, dirichlet=[], dgjumps=True) # space for velocity
    Phbase = H1(mesh, order=order, dirichlet=[], dgjumps=True) # space for pressure
    # U = Restrict(Uhbase, hasneg)
    U = Compress(Uhbase, GetDofsOfElements(Uhbase, ci.GetElementsOfType(HASNEG)))
    P = Compress(Phbase, GetDofsOfElements(Phbase, ci.GetElementsOfType(HASNEG)))
    # P = Restrict(Phbase, hasneg)
    fes = U*P
    (u,p), (v,q) = fes.TnT()
    gfu = GridFunction(fes)

    # Define special variables
    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))
    ne = specialcf.normal(2)
    strain_u = Sym(Grad(u))
    strain_v = Sym(Grad(v))

    # integration domains:
    dx = dCut(lsetp1, NEG, definedonelements=hasneg, deformation=deformation)
    ds = dCut(lsetp1, IF, definedonelements=hasif, deformation=deformation)
    dw = dFacetPatch(definedonelements=ba_facets, deformation=deformation)


    # 4. Construc bilinear form and right hand side 

    ah = BilinearForm(fes)
    # Au
    ah += 2*mu*InnerProduct(strain_u,strain_v)*dx + lam*div(u)*div(v)*dx \
            - (InnerProduct(Stress(strain_u)*n,v) + InnerProduct(Stress(strain_v)*n,u) - lambda_u/h*InnerProduct(u,v))*ds
    # order=1 i_s 
    ah += gamma_s * h * InnerProduct(Grad(u)*ne - Grad(u.Other())*ne,Grad(v)*ne - Grad(v.Other())*ne) * dw
    # -B
    ah += -alpha*(div(v)*p*dx  - p*v*n*ds)
    # Ap
    ah += K*grad(p)*grad(q)*dx \
            - (K*grad(p)*n*q + K*grad(q)*n*p - lambda_p/h*p*q)*ds
    # order=1 i_p 
    ah += gamma_p * h * (grad(p)*ne - grad(p.Other())*ne)*(grad(q)*ne - grad(q.Other())*ne) * dw
    # FPL stablization
    ah += tau_fpl*grad(p)*grad(q)*dx
    ah.Assemble()


    u_bar = GridFunction(U)
    p_bar = GridFunction(P)
    u_bar.Set(exact_u)
    p_bar.Set(exact_p)

    lh = LinearForm(fes)
    lh += b*v*dx - InnerProduct(u_bar,Stress(Sym(Grad(v)))*n)*ds + lambda_u/h*u_bar*v*ds #luh
    lh += f*q*dx - alpha*q*u_bar*n*ds - K*grad(q)*n*p_bar*ds + lambda_p/h*p_bar*q*ds #lph
    lh.Assemble()

    
    # 计算系数矩阵的条件数
    rows,cols,vals = ah.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    conds = np.linalg.cond(A.todense())
        
    # 5. Solve for the free dofs
    gfu.vec.data = ah.mat.Inverse() * lh.vec
    mask = IfPos(levelset,0,1)
    error_u = sqrt(Integrate(((gfu.components[0] - u_bar)*mask)**2, mesh))
    error_p = sqrt(Integrate((mask*(gfu.components[1] - p_bar))**2, mesh))

    return error_u,error_p, gfu.space.ndof, conds

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'Cond.':>12} | {'Error_u.':>12}  | {'Order':>6} | {'Error_p':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h,ndof,conds,error_u,error_p) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{error_u:12.4e} | {'-':>6}| {error_p:12.4e} | {'-':>6}")
        else:
            prev_h,_,_, prev_error_u,prev_error_p = results[i-1]
            rate_u = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{error_u:12.4e}| {rate_u:6.2f} | {error:12.4e} | {rate_p:6.2f}")



# Define important parameters
E = 1
nu = 0.2
mu  = E/2/(1+nu)
lam = E*nu/(1+nu)/(1-2*nu)
K = 0.1
alpha = 1
M = 100

quad_mesh = True
# Mesh diameter
h = 1/16
# Finite element space order
order = 1

# penalty parameters
lambda_u = 200*lam
lambda_p = 50*K
gamma_s = 20*lam
gamma_p = 10*K


# Manufactured exact solution for monitoring the error
u_x = sin(pi*x)*sin(pi*y)
u_y = sin(pi*x)*sin(pi*y)
exact_u = CF((u_x,u_y))
exact_p = cos(pi*y)+1

# strain tensor
epsilon_xx = u_x.Diff(x)
epsilon_yy = u_y.Diff(y) 
epsilon_xy = 0.5*(u_x.Diff(y) +  u_y.Diff(x))

# total stress tensor
sigma_xx = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_xx - alpha*exact_p
sigma_yy = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_yy - alpha*exact_p
sigma_xy = 2*mu*epsilon_xy

# 右端项 f_x, f_y
f_x = - (sigma_xx.Diff(x) + sigma_xy.Diff(y))
f_y = - (sigma_xy.Diff(x) + sigma_yy.Diff(y))

# 向量形式
b = CF( (f_x,f_y) ) # body force 
f = - K*(exact_p.Diff(x).Diff(x)+exact_p.Diff(y).Diff(y)) # source term

uD = exact_u
pD = exact_p


# Set level set function
levelset = sqrt(x**2 + y**2) - 0.5

results = []

for k in range(2, 6):
    h0 = 1/2**k
    tau_fpl = 1*h0**2
    error_u,error_p, gfu.space.ndof, conds = solve_biot_steady_unfitted(h0 quad_mesh, order, levelset, b, f, exact_u, exact_p, \
                                                                        mu,lam,tau_fpl,lambda_u,lambda_p,gamma_s,gamma_p,gamma_m,alpha,M,K):
    results.append((h0,ndof,conds,error_u,error_p))

print_convergence_table(results)
