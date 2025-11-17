from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
from math import pi,e
from numpy import linspace
import numpy as np
import scipy.sparse as sp


def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biotbrinkman_steady_unfitted(h0, quad_mesh, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, \
                               exact_p,alpha,K,mu,lam,beta_eta,beta_u,gamma_s,gamma_u,gamma_p,gamma_m,tau_p):

    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=2, threshold=0.1,
                                      discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1
    # InterpolateToP1(levelset,lsetp1)

    # Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    
    # facets used for stabilization:
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)
    ba_surround_facets = GetElementsWithNeighborFacets(mesh,ba_facets)
    interior_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg)
    in_surround_facets = GetElementsWithNeighborFacets(mesh,interior_facets)

    # 3. Construct the unfitted DG space 
    Ehbase = VectorL2(mesh, order=order_eta, dirichlet=[], dgjumps=True) # space for displacement
    Uhbase = VectorL2(mesh, order=order_u, dirichlet=[], dgjumps=True) # space for velocity
    Phbase = L2(mesh, order=order_p, dirichlet=[], dgjumps=True) # space for pressure 
    # Phbase = H1(mesh, order=order_p, dirichlet=[], dgjumps=True) # space for pressure 
    E = Compress(Ehbase, GetDofsOfElements(Ehbase, ci.GetElementsOfType(HASNEG)))
    U = Compress(Uhbase, GetDofsOfElements(Uhbase, ci.GetElementsOfType(HASNEG)))
    P = Compress(Phbase, GetDofsOfElements(Phbase, ci.GetElementsOfType(HASNEG)))
    fes = E*U*P
    (eta,u,p), (kxi,v,q) = fes.TnT()

    # Define special variables
    n = 1.0 / Norm(grad(lsetp1)) * grad(lsetp1)
    # n = specialcf.normal(2)
    ne = specialcf.normal(2) # normal vectors on faces
    h = specialcf.mesh_size  
    
    strain_eta = Sym(Grad(eta))
    strain_kxi = Sym(Grad(kxi))
    mean_stress_eta = 0.5*(Stress(Sym(Grad(eta)))+Stress(Sym(Grad(eta.Other()))))*ne
    mean_stress_kxi = 0.5*(Stress(Sym(Grad(kxi)))+Stress(Sym(Grad(kxi.Other()))))*ne
    jump_eta = eta - eta.Other()
    jump_kxi = kxi - kxi.Other()
    
    
    jump_p = p - p.Other()
    jump_q = q - q.Other()
    mean_p = 0.5*(p + p.Other())
    mean_q = 0.5*(q + q.Other())
    
    
    mean_dudn = 0.5*(Grad(u)+Grad(u.Other()))*ne
    mean_dvdn = 0.5*(Grad(v)+Grad(v.Other()))*ne
    jump_u = u - u.Other()
    jump_v = v - v.Other()

    # integration domains:
    domega = dCut(lsetp1, NEG, definedonelements=hasneg, deformation=deformation)
    dk = dCut(lsetp1, NEG, skeleton=True, definedonelements=interior_facets,
              deformation=deformation)
    ds = dCut(lsetp1, IF, definedonelements=hasif, deformation=deformation)
    dw = dFacetPatch(definedonelements=ba_facets, deformation=deformation)

    # 4. Construc bilinear form and right hand side 

    Ah = BilinearForm(fes)
    # Ae
    Ah += 2*mu*InnerProduct(strain_eta,strain_kxi)*domega + lam*div(eta)*div(kxi)*domega \
            - (InnerProduct(mean_stress_eta,jump_kxi) + InnerProduct(mean_stress_kxi,jump_eta) - beta_eta/h*InnerProduct(jump_eta,jump_kxi))*dk \
            - (InnerProduct(Stress(Sym(Grad(eta)))*n,kxi) + InnerProduct(Stress(Sym(Grad(kxi)))*n,eta) - beta_eta/h*InnerProduct(eta,kxi))*ds
    # order=1 i_s 
    Ah += gamma_s * h * InnerProduct(Grad(eta)*ne - Grad(eta.Other())*ne,Grad(kxi)*ne - Grad(kxi.Other())*ne) * dw
    
    # Be
    Ah += -alpha*(div(kxi)*p*domega - mean_p*jump_kxi*ne*dk - p*kxi*n*ds)
    
                  
    # Am
    Ah += nu*InnerProduct(Grad(u),Grad(v))*domega \
            - nu*(InnerProduct(mean_dudn,jump_v) + InnerProduct(mean_dvdn,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dk \
            - nu*(InnerProduct(Grad(u)*n,v) + InnerProduct(Grad(v)*n,u) - beta_u/h*InnerProduct(u,v))*ds\
            + K*InnerProduct(u,v)*domega
    # ghost penalty for velocity
    Ah += gamma_u * h * InnerProduct(Grad(u)*ne - Grad(u.Other())*ne,Grad(v)*ne - Grad(v.Other())*ne) * dw
    
    
     # Bm 
    Ah += -div(v)*p*domega + mean_p*jump_v*ne*dk + p*v*n*ds
    
    # -Bm
    Ah += div(u)*q*domega - mean_q*jump_u*ne*dk - q*u*n*ds
    
    # order=1 i_p 
    Ah += gamma_p * h * (grad(p)*ne - grad(p.Other())*ne)*(grad(q)*ne - grad(q.Other())*ne) * dw
    
    # M
    Ah += s0*p*q*domega + gamma_m*(h**3)*(grad(p)*ne - grad(p.Other())*ne)*(grad(q)*ne - grad(q.Other())*ne) * dw
    
    # -Be
    Ah += alpha*(div(eta)*q*domega - mean_q*jump_eta*ne*dk - q*eta*n*ds)
    
    # stabilization
    Ah += tau_p*h*jump_p*jump_q*dk
    Ah.Assemble()


    # r.h.s
    lh = LinearForm(fes) 
    lh += fe*kxi*domega - InnerProduct(etaD,Stress(Sym(Grad(kxi)))*n)*ds + beta_eta/h*etaD*kxi*ds
    lh += fm*v*domega - nu*InnerProduct(uD,Grad(v)*n)*ds + nu*beta_u/h*uD*v*ds
    lh += fp*q*domega - alpha*q*etaD*n*ds - q*uD*n*ds
    lh.Assemble()

    
    # # 计算系数矩阵的条件数
    # rows,cols,vals = ah.mat.COO()
    # A = sp.csr_matrix((vals,(rows,cols)))
    # conds = np.linalg.cond(A.todense())
        
    # 5. Solve for the free dofs
    gfu = GridFunction(fes)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec
    
    error_eta = sqrt(Integrate((gfu.components[0] - exact_eta)**2* domega, mesh))
    error_u = sqrt(Integrate((gfu.components[1] - exact_u)**2 * domega, mesh))
    error_p = sqrt(Integrate((gfu.components[2] - exact_p)**2 * domega, mesh))
    

    return error_eta, error_u,error_p, gfu.space.ndof

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} |  {'Error_eta.':>12} | {'Order':>6} | {'Error_u.':>12}  | {'Order':>6} | {'Error_p':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h,dofs,error_eta,error_u,error_p) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {'-':>6} |{error_u:12.4e}| {'-':>6}| {error_p:12.4e} | {'-':>6}")
        else:
            prev_h,_,prev_error_eta, prev_error_u,prev_error_p = results[i-1]
            rate_eta = (np.log(prev_error_eta) - np.log(error_eta)) / (np.log(prev_h) - np.log(h))
            rate_u = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {rate_eta:6.2f} |{error_u:12.4e}| {rate_u:6.2f} | {error_p:12.4e} | {rate_p:6.2f}")



# Define important parameters
mu  = 10
lam = 10
alpha = 1
# alpha = 0
K = 1 # k^-1
nu = 1
# s0 = 10
s0 = 1e-5

quad_mesh = False

# DG space order
order_eta = 2
order_u = 2
order_p = 1

# penalty parameters
# p2-p2-p1 (200,200,0,0,1,0,0.1)
beta_eta = 400
beta_u = 200
# ghost penalty parameters
gamma_s = 0
gamma_u = 0
gamma_p = 0
gamma_m = 0
tau_p = 0.1

# Manufactured exact solution for monitoring the error
u_x = sin(pi*x) * cos(pi*y)
u_y = -cos(pi*x) * sin(pi*y)
eta_x = sin(pi*x)*sin(pi*y)
eta_y = x*y*(x-1)*(y-1)

exact_eta = CF((eta_x, eta_y))
exact_u = CF((u_x,u_y))
exact_p = sin(pi*(x-y))



# strain tensor
epsilon_xx = eta_x.Diff(x)
epsilon_yy = eta_y.Diff(y) 
epsilon_xy = 0.5*(eta_x.Diff(y) +  eta_y.Diff(x))

# total stress tensor
sigma_xx = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_xx - alpha*exact_p
sigma_yy = lam*(epsilon_xx + epsilon_yy) + 2*mu*epsilon_yy - alpha*exact_p
sigma_xy = 2*mu*epsilon_xy

# 右端项 f_x, f_y
f_x = - (sigma_xx.Diff(x) + sigma_xy.Diff(y))
f_y = - (sigma_xy.Diff(x) + sigma_yy.Diff(y))

fm_x = K*u_x-nu*(u_x.Diff(x).Diff(x)+u_x.Diff(y).Diff(y)) + exact_p.Diff(x)
fm_y = K*u_y-nu*(u_y.Diff(x).Diff(x)+u_y.Diff(y).Diff(y)) + exact_p.Diff(y)

# 向量形式
fe = CF((f_x, f_y))
fm = CF((fm_x, fm_y))
fp = s0*exact_p+alpha*(eta_x.Diff(x)+eta_y.Diff(y)) + u_x.Diff(x) + u_y.Diff(y)

etaD = exact_eta
uD = exact_u
pD = exact_p

# Set level set function
levelset = sqrt(x**2 + y**2) - 0.5

results = []

for k in range(2, 6):
    h0 = 1/2**k
    error_eta, error_u,error_p,ndof = solve_biotbrinkman_steady_unfitted(h0, quad_mesh, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, \
                                                              exact_p,alpha,K,mu,lam,beta_eta,beta_u,gamma_s,gamma_u,gamma_p,gamma_m,tau_p)
    results.append((h0,ndof,error_eta,error_u,error_p))

print_convergence_table(results)
