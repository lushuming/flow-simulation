from netgen.occ import *
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.webgui import Draw
from math import pi,e
from numpy import linspace

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biotbrinkman_dg(dt, h0, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, mu,lam,beta_eta,beta_u,alpha,K):
    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((0, 0), (1, 1),bc=1)
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=False)
    mesh = Mesh(ngmesh)

    # 2. Define the DG space 
    # DG spaces
    E = VectorL2(mesh, order=order_eta, dirichlet=[], dgjumps=True) # space for displacement
    U = VectorL2(mesh, order=order_u, dirichlet=[], dgjumps=True) # space for velocity
    P = L2(mesh, order=order_p, dirichlet=[], dgjumps=True) # space for pressure  
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
   
    # 4. Define bilinear form
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
    
    Ah.Assemble()
    
    Mh = BilinearForm(fes)
    # M
    Mh += s0*p*q*dx
    
    # -Be
    Mh += alpha*(div(eta)*q*dx - mean_q*jump_eta*n*dx(skeleton=True) - q*eta*n*ds(skeleton=True))
    Mh.Assemble()
    
    mstar = Mh.mat.CreateMatrix()
    # corresponds to M* = M/dt + A
    mstar.AsVector().data = 1/dt*Mh.mat.AsVector() + Ah.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs())

    # r.h.s
    lh = LinearForm(fes) 
    lh += fe*kxi*dx - InnerProduct(etaD,Stress(Sym(Grad(kxi)))*n)*ds(skeleton=True) + beta_eta/h*etaD*kxi*ds(skeleton=True)
    lh += fm*v*dx - nu*InnerProduct(uD,Grad(v)*n)*ds(skeleton=True) + nu*beta_u/h*uD*v*ds(skeleton=True)
    lh += fp*q*dx - alpha*q*etaD*n*ds(skeleton=True) - q*uD*n*ds(skeleton=True)

    return fes,invmstar,lh,Mh,mesh



def TimeStepping(invmstar, initial_condeta = None, initial_condu = None, initial_condp = None, t0 = 0, tend = 1,
                      nsamples = 10):
    if initial_condu and initial_condp :
        gfu.components[0].Set(initial_condeta)
        gfu.components[1].Set(initial_condu)
        gfu.components[2].Set(initial_condp)
    cnt = 0; # 时间步计数
    time = t0 # 当前时间
    sample_int = int(floor(tend / dt / nsamples)+1) # 采样间隔，用于决定什么时候把解存入 gfut
    gfut = GridFunction(gfu.space,multidim=0) #存储所有采样时间步的结果，多维 GridFunction
    gfut.AddMultiDimComponent(gfu.vec)
    while time <  tend + 1e-7:
        t.Set(time+dt)
        lh.Assemble()
        res = lh.vec + 1/dt * Mh.mat * gfu.vec
        gfu.vec.data = invmstar * res
        print("\r",time,end="")
        # print(time,end="\n")
        if cnt % sample_int == 0:
            gfut.AddMultiDimComponent(gfu.vec)
        cnt += 1; time = cnt * dt
    return gfut,gfu

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



# penalty parameters
beta_eta = 200
beta_u = 200
# gamma_p2 = 1
# physical parameters
mu  = 10
lam = 10
# alpha = 1
alpha = 1
K = 1 # k^-1
nu = 1
s0 = 1e-2

# set up the analytical solution
t = Parameter(0.0) # A Parameter is a constant CoefficientFunction the value of which can be changed with the Set-function.

# eta_x = sin(pi*x)**2 * sin(pi*y)
# eta_y = -sin(pi*x) * sin(pi*y)**2
u_x = e**(-t)*sin(pi*x) * cos(pi*y)
u_y = -e**(-t)*cos(pi*x) * sin(pi*y)
# exact_p =cos(pi*x) * cos(pi*y) + 2


eta_x = e**(-t)*sin(pi*x)*sin(pi*y)
eta_y = e**(-t)*x*y*(x-1)*(y-1)
# u_x = 4*(x**3)*(y**4)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**3)*((y-1)**4)
# u_y = 4*(x**4)*(y**3)*((x-1)**4)*((y-1)**4) + 4*(x**4)*(y**4)*((x-1)**4)*((y-1)**3)
exact_eta = CF((eta_x, eta_y))
exact_u = CF((u_x,u_y))
exact_p = e**(-t)*sin(pi*(x-y))
# exact_p = sin(pi*x)*sin(pi*y)

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
fp = (s0*exact_p+alpha*(eta_x.Diff(x)+eta_y.Diff(y))).Diff(t) + u_x.Diff(x) + u_y.Diff(y)

etaD = exact_eta
uD = exact_u
pD = exact_p


order_eta = 2
order_u = 2
order_p = 1

results = []

for k in range(2, 6):
    h0 = 1/2**k
    fes,invmstar,lh,Mh,mesh = solve_biotbrinkman_dg(dt, h0, order_eta, order_u,order_p, fe, fm,fp, exact_eta, exact_u, exact_p, mu,lam,beta_eta,beta_u,alpha,K)
    gfu = GridFunction(fes)
    gfut,gfu = TimeStepping(invmstar,lh, Mh,initial_condeta = exact_eta, initial_condu=exact_u, initial_condp=exact_p, tend=endT, nsamples=20)
    error_eta = sqrt(Integrate((gfu.components[0] - exact_eta)**2, mesh))
    error_u = sqrt(Integrate((gfu.components[0] - exact_u)**2, mesh))
    error_p = sqrt(Integrate((gfu.components[1] - exact_p)**2, mesh))
    ndof = gfu.space.ndof
    results.append((h0,ndof,error_eta, error_u, error_p ))

print_convergence_table(results)
