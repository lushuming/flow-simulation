from netgen.occ import *
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.webgui import Draw
from math import pi,e
from numpy import linspace
import numpy as np


def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biotBrinkman_steady_dg(h0, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, mu,lam,beta_eta,beta_u,gamma_p,alpha,K):

    # 1. Construct the mesh
    # mesh = Mesh(unit_square.GenerateMesh(maxh=h))
    square = SplineGeometry()
    # square.AddRectangle((0, 0), (1, 1),bc=1)
    
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=False)
    mesh = Mesh(ngmesh)

    E = VectorL2(mesh, order=order_eta, dirichlet=[], dgjumps=True) # space for displacement
    U = VectorL2(mesh, order=order_u, dirichlet=[], dgjumps=True) # space for velocity
    # P = L2(mesh, order=order_p, dirichlet=[], dgjumps=True) # space for pressure
    P = H1(mesh, order=order_p, dirichlet=[], dgjumps=True) # space for pressure 
    fes = E*U*P
    (eta,u,p), (kxi,v,q) = fes.TnT()
    
    # Define the jumps and the averages
    n = specialcf.normal(2)
    h = specialcf.mesh_size  
    
    strain_eta = Sym(Grad(eta))
    strain_kxi = Sym(Grad(kxi))
    mean_stress_eta = 0.5*(Stress(Sym(Grad(eta)))+Stress(Sym(Grad(eta.Other()))))*n
    mean_stress_kxi = 0.5*(Stress(Sym(Grad(kxi)))+Stress(Sym(Grad(kxi.Other()))))*n
    jump_eta = eta - eta.Other()
    jump_kxi = kxi - kxi.Other()
    
    
    jump_p = p - p.Other()
    jump_q = q - q.Other()
    mean_p = 0.5*(p + p.Other())
    mean_q = 0.5*(q + q.Other())
    
    
    mean_dudn = 0.5*(Grad(u)+Grad(u.Other()))*n
    mean_dvdn = 0.5*(Grad(v)+Grad(v.Other()))*n
    jump_u = u - u.Other()
    jump_v = v - v.Other()

    Ah = BilinearForm(fes)
    # Ae
    Ah += 2*mu*InnerProduct(strain_eta,strain_kxi)*dx + lam*div(eta)*div(kxi)*dx \
            - (InnerProduct(mean_stress_eta,jump_kxi) + InnerProduct(mean_stress_kxi,jump_eta) - beta_eta/h*InnerProduct(jump_eta,jump_kxi))*dx(skeleton=True) \
            - (InnerProduct(Stress(Sym(Grad(eta)))*n,kxi) + InnerProduct(Stress(Sym(Grad(kxi)))*n,eta) - beta_eta/h*InnerProduct(eta,kxi))*ds(skeleton=True)

    # Be
    Ah += -alpha*(div(kxi)*p*dx - mean_p*jump_kxi*n*dx(skeleton=True) - p*kxi*n*ds(skeleton=True))

    # Am
    Ah += nu*InnerProduct(Grad(u),Grad(v))*dx \
            - nu*(InnerProduct(mean_dudn,jump_v) + InnerProduct(mean_dvdn,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dx(skeleton=True) \
            - nu*(InnerProduct(Grad(u)*n,v) + InnerProduct(Grad(v)*n,u) - beta_u/h*InnerProduct(u,v))*ds(skeleton=True) \
            + K*InnerProduct(u,v)*dx

    # Bm 
    Ah += -div(v)*p*dx + mean_p*jump_v*n*dx(skeleton=True) + p*v*n*ds(skeleton=True)

    # -Bm
    Ah += div(u)*q*dx - mean_q*jump_u*n*dx(skeleton=True) - q*u*n*ds(skeleton=True)

    # M
    Ah += s0*p*q*dx

    # -Be
    Ah += alpha*(div(eta)*q*dx - mean_q*jump_eta*n*dx(skeleton=True) - q*eta*n*ds(skeleton=True))

    # Ah += gamma_p*h*h*grad(p)*grad(q)*dx
    Ah += gamma_p*h*jump_p*jump_q*dx(skeleton=True) 
    Ah.Assemble()

    lh = LinearForm(fes)
    # lh += fe*kxi*dx - InnerProduct(etaD,Stress(Sym(Grad(kxi)))*n)*ds(skeleton=True) + beta_eta/h*etaD*kxi*ds(skeleton=True) + alpha*pD*kxi*n*ds(skeleton=True)
    lh += fe*kxi*dx - InnerProduct(etaD,Stress(Sym(Grad(kxi)))*n)*ds(skeleton=True) + beta_eta/h*etaD*kxi*ds(skeleton=True)
    # lh += fm*v*dx - nu*InnerProduct(uD,Grad(v)*n)*ds(skeleton=True) + nu*beta_u/h*uD*v*ds(skeleton=True) + v*n*pD*ds(skeleton=True) 
    lh += fm*v*dx - nu*InnerProduct(uD,Grad(v)*n)*ds(skeleton=True) + nu*beta_u/h*uD*v*ds(skeleton=True)
    lh += fp*q*dx - alpha*q*etaD*n*ds(skeleton=True) - q*uD*n*ds(skeleton=True)
    lh.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec
    
    error_eta = sqrt(Integrate((gfu.components[0] - exact_eta)**2, mesh))
    error_u = sqrt(Integrate((gfu.components[1] - exact_u)**2, mesh))
    error_p = sqrt(Integrate((gfu.components[2] - exact_p)**2, mesh))

    gff = GridFunction(fes)
    gff.components[0].Set(exact_eta)
    gff.components[1].Set(exact_u)

    grad_error_eta = Grad(gfu.components[0] - gff.components[0])
    grad_error_u = Grad(gfu.components[1] - gff.components[1])

    error_eta_H1 = sqrt(Integrate(InnerProduct(grad_error_eta,grad_error_eta)* dx, mesh))
    error_u_H1 = sqrt(Integrate(InnerProduct(grad_error_u,grad_error_u)*  dx, mesh))
        
    return error_eta, error_u, error_p, gfu.space.ndof, error_eta_H1,error_u_H1

# def print_convergence_table(results):
#     print(f"{'h':>8} | {'DoFs':>8} |  {'Error_eta.':>12} | {'Order':>6} | {'Error_u.':>12}  | {'Order':>6} | {'Error_p':>12} | {'Order':>6}")
#     print("-" * 70)
#     for i, (h,dofs,error_eta,error_u,error_p) in enumerate(results):
#         if i == 0:
#             print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {'-':>6} |{error_u:12.4e}| {'-':>6}| {error_p:12.4e} | {'-':>6}")
#         else:
#             prev_h,_,prev_error_eta, prev_error_u,prev_error_p = results[i-1]
#             rate_eta = (np.log(prev_error_eta) - np.log(error_eta)) / (np.log(prev_h) - np.log(h))
#             rate_u = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
#             rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
#             print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {rate_eta:6.2f} |{error_u:12.4e}| {rate_u:6.2f} | {error_p:12.4e} | {rate_p:6.2f}")

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} |  {'Error_eta_L2.':>12} | {'Order':>6} |  {'Error_eta_H1.':>12} | {'Order':>6}  | {'Error_u_L2.':>12}  | {'Order':>6} |{'Error_u_H1.':>12}  | {'Order':>6}| {'Error_p':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h,dofs,error_eta,error_eta_H1,error_u,error_u_H1,error_p) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} |  {'-':>6} | {error_eta_H1:12.4e} |  {'-':>6} | {error_u:12.4e}| {'-':>6}| {error_u_H1:12.4e}| {'-':>6}  | {error_p:12.4e} | {'-':>6}")
        else:
            prev_h,_,prev_error_eta,prev_error_eta_H1, prev_error_u,prev_error_u_H1,prev_error_p = results[i-1]
            rate_eta_L2 = (np.log(prev_error_eta) - np.log(error_eta)) / (np.log(prev_h) - np.log(h))
            rate_eta_H1 = (np.log(prev_error_eta_H1) - np.log(error_eta_H1)) / (np.log(prev_h) - np.log(h))
            rate_u_L2 = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
            rate_u_H1 = (np.log(prev_error_u_H1) - np.log(error_u_H1)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {rate_eta_L2:6.2f} |{error_eta_H1:12.4e} | {rate_eta_H1:6.2f}|{error_u:12.4e}| {rate_u_L2:6.2f} | {error_u_H1:12.4e}| {rate_u_H1:6.2f} | {error_p:12.4e} | {rate_p:6.2f}")


# 10, 100, 1, 10, 1, 0.01, P1 x P1 x P0, 
# 10, 100, 1, 10, 1, 0.01, P2 x P2 x P1, 200, 100, 20, 10, 0.1, 0, 0


# Define important parameters
# mu = 10, lambda = 10  (无 locking 现象)
mu  = 10
lam = 100
alpha = 1
# alpha = 0
K = 10 # k^-1
nu = 1
# s0 = 10
s0 = 1e-2


quad_mesh = False
# Finite element space order
order_eta = 2
order_u = 2
order_p = 1

# penalty parameters
# p2-p2-p1 beta_eta = 200, beta_u = 100
# p3-p3-p2 beta_eta = 300, beta_u = 300
beta_eta = 100
beta_u = 100
gamma_p = 0

# Manufactured exact solution for monitoring the error
#---------------------Example 1 -----------------------
# eta_x = sin(pi*x)*sin(pi*y)
# eta_y = x*y*(x-1)*(y-1)
# u_x = sin(pi*x) * cos(pi*y)
# u_y = -cos(pi*x) * sin(pi*y)
# exact_p = sin(pi*(x-y))

#---------------------Example 2 -----------------------
# eta_x = sin(pi*x)*sin(pi*y) + x/lam
# eta_y = cos(pi*x)*cos(pi*y) + y/lam
# # eta_x = -x*x*y*(2*y-1)*(x-1)*(x-1)*(y-1)
# # eta_y = x*y*y*(2*x-1)*(x-1)*(y-1)*(y-1)
# u_x = x*x*y*y+exp(-y)
# u_y = -2/3*x*y**3+2-pi*sin(pi*x)
# exact_p = (pi*sin(pi*x)-2)*cos(2*pi*y)

#---------------------Example 3 -----------------------
u_x = sin(pi*x) * cos(pi*y)
u_y = -cos(pi*x) * sin(pi*y)
eta_x = sin(pi*x)*sin(pi*y) 
eta_y = cos(pi*x)*cos(pi*y) 
exact_p = sin(pi*(x-y))


# eta_x = sin(pi*x)**2 * sin(pi*y)
# eta_y = -sin(pi*x) * sin(pi*y)**2
# u_x = 4*(x**3)*(y**4)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**3)*((y-1)**4)
# u_y = 4*(x**4)*(y**3)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**4)*((y-1)**3)
# exact_p =cos(pi*x) * cos(pi*y) + 2

exact_eta = CF((eta_x, eta_y))
exact_u = CF((u_x,u_y))

# exact_p = eta_x = sin(pi*x)*sin(pi*y)

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

results = []

for k in range(2, 6):
    h0 = 1/2**k
    error_eta, error_u, error_p, ndof, error_eta_H1,error_u_H1 = solve_biotBrinkman_steady_dg(h0, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, \
                                                                               mu,lam,beta_eta,beta_u,gamma_p,alpha,K)
    # results.append((h0,ndof,error_eta,error_u,error_p))
    results.append((h0,ndof,error_eta,error_eta_H1,error_u,error_u_H1,error_p))

print_convergence_table(results)
