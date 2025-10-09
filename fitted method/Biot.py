from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
from math import pi,e
from numpy import linspace
import numpy as np

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biot_dg(dt, F, g, exact_u, exact_p, beta_u=10, beta_p=10, gamma_p=10, order=1, mh=0.1):
    # 1. Construct the mesh
    mesh = Mesh(unit_square.GenerateMesh(maxh=mh))

    # 2. Define the DG space 
    U = VectorL2(mesh, order=order, dirichlet=".*", dgjumps=True) # space for velocity
    P = L2(mesh, order=order, dirichlet=".*", dgjumps=True) # space for pressure
    fes = U*P
    (u,p), (v,q) = fes.TnT()
    gfu = GridFunction(fes)
    
    # 3. Define the jumps and the averages
    jump_u = u - u.Other()
    jump_v = v - v.Other()
    jump_p = p - p.Other()
    jump_q = q - q.Other()
    n = specialcf.normal(2)
    strain_u = Sym(Grad(u))
    strain_v = Sym(Grad(v))
    mean_stress_u = 0.5*(Stress(Sym(Grad(u)))+Stress(Sym(Grad(u.Other()))))*n
    mean_stress_v = 0.5*(Stress(Sym(Grad(v)))+Stress(Sym(Grad(v.Other()))))*n
    mean_dpdn = 0.5*K*(grad(p)+grad(p.Other()))*n
    mean_dqdn = 0.5*K*(grad(q)+grad(q.Other()))*n
    mean_p = 0.5*(p + p.Other())
    mean_q = 0.5*(q + q.Other())
    h = specialcf.mesh_size   
   
    # 4. Define bilinear form
    ah = BilinearForm(fes)
    # Au
    ah += 2*mu*InnerProduct(strain_u,strain_v)*dx + lam*div(u)*div(v)*dx \
            - (InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dx(skeleton=True) \
            - (InnerProduct(Stress(Sym(Grad(u)))*n,v) + InnerProduct(Stress(Sym(Grad(v)))*n,u) - beta_u/h*InnerProduct(u,v))*ds(skeleton=True)
    # -B
    ah += -alpha*(div(v)*p*dx - mean_p*jump_v*n*dx(skeleton=True) - p*v*n*ds(skeleton=True))
    # Ap
    ah += K*grad(p)*grad(q)*dx \
            - (mean_dpdn*jump_q + mean_dqdn*jump_p - beta_p/h*jump_p*jump_q)*dx(skeleton=True) \
            - (K*grad(p)*n*q + K*grad(q)*n*p - beta_p/h*p*q)*ds(skeleton=True) 
    ah.Assemble()
    
    mh = BilinearForm(fes)
    # C
    mh += c0*p*q*dx + gamma_p*h*h*grad(p)*grad(q)*dx
    # B^T
    mh += alpha*(div(u)*q*dx - mean_q*jump_u*n*dx(skeleton=True) - q*u*n*ds(skeleton=True))
    mh.Assemble()
    
    mstar = mh.mat.CreateMatrix()
    # corresponds to M* = M/dt + A
    mstar.AsVector().data = 1/dt*mh.mat.AsVector() + ah.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs())

    # r.h.s
    f = LinearForm(fes)
    f += F*v*dx - InnerProduct(uD,Stress(Sym(Grad(v)))*n)*ds(skeleton=True) + beta_u/h*uD*v*ds(skeleton=True)
    f += g*q*dx - alpha*q*uD*n*ds(skeleton=True) - K*grad(q)*n*pD*ds(skeleton=True) + beta_p/h*pD*q*ds(skeleton=True)

    return fes,invmstar,f,mh,mesh



def TimeStepping(invmstar, f, mh, initial_condu = None, initial_condp = None, t0 = 0, tend = 1,
                      nsamples = 10):
    if initial_condu and initial_condp :
        gfu.components[0].Set(initial_condu)
        gfu.components[1].Set(initial_condp)
    cnt = 0; # 时间步计数
    time = t0 # 当前时间
    sample_int = int(floor(tend / dt / nsamples)+1) # 采样间隔，用于决定什么时候把解存入 gfut
    gfut = GridFunction(gfu.space,multidim=0) #存储所有采样时间步的结果，多维 GridFunction
    gfut.AddMultiDimComponent(gfu.vec)
    while time <  tend + 1e-7:
        t.Set(time)
        f.Assemble()
        res = f.vec + 1/dt * mh.mat * gfu.vec
        gfu.vec.data = invmstar * res
        print("\r",time,end="")
        # print(time,end="\n")
        if cnt % sample_int == 0:
            gfut.AddMultiDimComponent(gfu.vec)
        cnt += 1; time = cnt * dt
    return gfut,gfu

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'u_L2 Error':>12} | {'Order':>6} | {'p_L2 Error':>12} | {'Order':>6}")
    print("-" * 60)
    for i, (h0, dofs, erroru,errorp) in enumerate(results):
        if i == 0:
            print(f"{h0:8.4f} | {dofs:8d} | {erroru:12.4e} | {'-':>6}| {errorp:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_erroru,prev_errorp = results[i-1]
            rate_u = (np.log(prev_erroru) - np.log(erroru)) / (np.log(prev_h) - np.log(h0))
            rate_p = (np.log(prev_errorp) - np.log(errorp)) / (np.log(prev_h) - np.log(h0))
            print(f"{h0:8.4f} | {dofs:8d} | {erroru:12.4e} | {rate_u:6.2f}| {errorp:12.4e} | {rate_p:6.2f}")



# Define important parameters
order = 1
endT = 1
dt = 0.01
# penalty parameters
beta_u = 100 
beta_p = 100
# stablization paramer
gamma_p = 1

# physical parameters
mu  = 1
lam = 1
c0 = 1e-4
alpha = 1
K = 1

# set up the analytical solution
t = Parameter(0.0) # A Parameter is a constant CoefficientFunction the value of which can be changed with the Set-function.
u_x = e**(-t)*(sin(2*pi*y)*(-1+cos(2*pi*x))+sin(pi*x)*sin(pi*y)/(lam+mu))
u_y = e**(-t)*(sin(2*pi*x)*(1-cos(2*pi*y))+sin(pi*x)*sin(pi*y)/(lam+mu))
exact_u = CF((u_x,u_y))
exact_p = e**(-t)*sin(pi*x)*sin(pi*y)

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
F = CF( (f_x,f_y) )
g = (c0*exact_p+alpha*(u_x.Diff(x)+u_y.Diff(y))).Diff(t) - K*(exact_p.Diff(x).Diff(x)+exact_p.Diff(y).Diff(y))

uD = exact_u
pD = exact_p

results = []

for k in range(2, 5):
    h0 = 1/2**k
    fes,invmstar,f,mh,mesh = solve_biot_dg(dt, F, g, exact_u, exact_p, beta_u, beta_p, gamma_p, order, h0)
    gfu = GridFunction(fes)
    gfut,gfu = TimeStepping(invmstar,f, mh, initial_condu=exact_u, initial_condp=exact_p, tend=endT, nsamples=20)
    error_u = sqrt(Integrate((gfu.components[0] - exact_u)**2, mesh))
    error_p = sqrt(Integrate((gfu.components[1] - exact_p)**2, mesh))
    ndof = gfu.space.ndof
    results.append((h0,ndof,error_u, error_p ))

print_convergence_table(results)
