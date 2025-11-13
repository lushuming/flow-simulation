from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
from math import pi,e
from numpy import linspace


def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biotBrinkman_steady_dg(h, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, mu,lam,beta_eta,beta_u,alpha,K):

    # 1. Construct the mesh
    mesh = Mesh(unit_square.GenerateMesh(maxh=h))

    E = VectorL2(mesh, order=order_eta, dirichlet=".*", dgjumps=True) # space for displacement
    U = VectorL2(mesh, order=order_u, dirichlet=".*", dgjumps=True) # space for velocity
    P = L2(mesh, order=order_p, dirichlet=".*", dgjumps=True) # space for pressure
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
            - (InnerProduct(Stress(Sym(Grad(eta)))*n,kxi) + InnerProduct(Stress(Sym(Grad(kxi)))*n,eta) - beta_eta/h*InnerProduct(kxi,eta))*ds(skeleton=True)
    
    # Be
    Ah += -alpha*(div(kxi)*p*dx - mean_p*jump_kxi*n*dx(skeleton=True) - p*kxi*n*ds(skeleton=True))
    
    # Am
    Ah += 2*nu*InnerProduct(Grad(u),Grad(v))*dx \
            - 2*nu*(InnerProduct(mean_dudn,jump_v) + InnerProduct(mean_dvdn,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dx(skeleton=True) \
            - 2*nu*(InnerProduct(Grad(u)*n,v) + InnerProduct(Grad(v)*n,u) - beta_u/h*InnerProduct(u,v))*ds(skeleton=True) \
            + K*InnerProduct(u,v)*dx
    
    # Bm 
    Ah += -div(v)*p*dx + mean_p*jump_v*n*dx(skeleton=True) + p*v*n*ds(skeleton=True)
    
    # -Bm
    Ah += div(u)*q*dx - mean_q*jump_u*n*dx(skeleton=True) - q*u*n*ds(skeleton=True)
    
    # M
    Ah += s0*p*q*dx
    
    # -Be
    Ah += alpha*(div(eta)*q*dx - mean_q*jump_eta*n*dx(skeleton=True) - q*eta*n*ds(skeleton=True))
    
    Ah.Assemble()

    lh = LinearForm(fes)
    lh += fe*kxi*dx - InnerProduct(etaD,Stress(Sym(Grad(kxi)))*n)*ds(skeleton=True) + beta_eta/h*etaD*kxi*ds(skeleton=True)
    lh += fm*v*dx - 2*nu*InnerProduct(uD,Stress(Sym(Grad(v)))*n)*ds(skeleton=True) + 2*nu*beta_u/h*uD*v*ds(skeleton=True) + v*n*pD*ds(skeleton=True) 
    lh += fp*q*dx 
    lh.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec
    
    error_eta = sqrt(Integrate((gfu.components[0] - exact_eta)**2, mesh))
    error_u = sqrt(Integrate((gfu.components[1] - exact_u)**2, mesh))
    error_p = sqrt(Integrate((gfu.components[2] - exact_p)**2, mesh))
        
    return error_eta, error_u, error_p, gfu.space.ndof

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} |  {'Error_eta.':>12} | {'Order':>6} | {'Error_u.':>12}  | {'Order':>6} | {'Error_p':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h,dofs,error_eta,error_u,error_p) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} |  {'-':>6} |{error_u:12.4e} | {'-':>6}| {error_p:12.4e} | {'-':>6}")
        else:
            prev_h,_,prev_error_eta, prev_error_u,prev_error_p = results[i-1]
            rate_eta = (np.log(prev_error_eta) - np.log(error_eta)) / (np.log(prev_h) - np.log(h))
            rate_u = (np.log(prev_error_u) - np.log(error_u)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {error_eta:12.4e} | {rate_eta:6.2f} |{error_u:12.4e}| {rate_u:6.2f} | {error_p:12.4e} | {rate_p:6.2f}")



# Define important parameters
mu  = 10
lam = 100
alpha = 1
K = 10 # k^-1
nu = 1
s0 = 1e-2


quad_mesh = True
# Finite element space order
order_eta = 1
order_u = 1
order_p = 0

# penalty parameters
beta_eta = 100
beta_u = 100 

# Manufactured exact solution for monitoring the error
eta_x = sin(pi*x)*sin(pi*y)
eta_y = x*y*(x-1)*(y-1)
u_x = 4*(x**3)*(y**4)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**3)*((y-1)**4)
u_y = 4*(x**4)*(y**3)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**4)*((y-1)**3)
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

fm_x = K*u_x-2*nu*(u_x.Diff(x).Diff(x)+u_x.Diff(y).Diff(y)) + exact_p.Diff(x)
fm_y = K*u_y-2*nu*(u_y.Diff(x).Diff(x)+u_y.Diff(y).Diff(y)) + exact_p.Diff(y)

# 向量形式
fe = CF((f_x, f_y))
fm = CF((fm_x, fm_x))
# fp = (s0*exact_p+alpha*(eta_x.Diff(x)+eta_y.Diff(y))).Diff(t) + u_x.Diff(x) + u_y.Diff(y)
fp = s0*exact_p+alpha*(eta_x.Diff(x)+eta_y.Diff(y)) + u_x.Diff(x) + u_y.Diff(y)

etaD = exact_eta
uD = exact_u
pD = exact_p

results = []

for k in range(2, 6):
    h0 = 1/2**k
    error_eta, error_u, error_p, ndof = solve_biotBrinkman_steady_dg(h, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, \
                                                                               mu,lam,beta_eta,beta_u,alpha,K)
    results.append((h0,ndof,error_eta,error_u,error_p))

print_convergence_table(results)
