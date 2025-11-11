from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
from math import pi,e
from numpy import linspace
import numpy as np

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_biot_cutfem(dt, b, f, uD, pD, levelset, quad_mesh, mu,lam,tau_fpl,lambda_u,lambda_p,gamma_s,gamma_p,gamma_m,alpha,M,K, orderu, orderp, h=0.1):
    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=h, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2. Get lset
    # Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=1, threshold=0.1, discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1
    InterpolateToP1(levelset,lsetp1)

    # Cut information
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)

    # 3. Define the unfitted fem space 
    Uhbase = VectorH1(mesh, order=orderu, dirichlet=[], dgjumps=True) # space for velocity
    Phbase = H1(mesh, order=orderp, dirichlet=[], dgjumps=True) # space for pressure
    U = Compress(Uhbase, GetDofsOfElements(Uhbase, ci.GetElementsOfType(HASNEG)))
    P = Compress(Phbase, GetDofsOfElements(Phbase, ci.GetElementsOfType(HASNEG)))
    fes = U*P
    (u,p), (v,q) = fes.TnT()
    gfu = GridFunction(fes)

    # uD = GridFunction(U)
    # pD = GridFunction(P)
    # uD.Set(exact_u)
    # pD.Set(exact_p)

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
    
    # 4. Define bilinear form
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

    # ah += 1/dt*(1/M*p*q*dx + gamma_m*(h**3)*(grad(p)*ne - grad(p.Other())*ne)*(grad(q)*ne - grad(q.Other())*ne) * dw)
    # ah += 1/dt*alpha*(div(u)*q*dx - q*u*n*ds)

    ah.Assemble()
    
    mh = BilinearForm(fes)
    # C
    mh += 1/M*p*q*dx + gamma_m*(h**3)*(grad(p)*ne - grad(p.Other())*ne)*(grad(q)*ne - grad(q.Other())*ne) * dw
    # B^T
    mh += alpha*(div(u)*q*dx - q*u*n*ds)
    # mh += alpha*div(u)*q*dx
    mh.Assemble()
    
    mstar = mh.mat.CreateMatrix()
    # corresponds to M* = M/dt + A
    mstar.AsVector().data = (1/dt)*mh.mat.AsVector() + ah.mat.AsVector()
    invmstar = mstar.Inverse(freedofs=fes.FreeDofs())

    # r.h.s
    lh = LinearForm(fes)
    lh += b*v*dx - InnerProduct(uD,Stress(Sym(Grad(v)))*n)*ds + lambda_u/h*uD*v*ds
    # lh += f*q*dx - 1/dt*alpha*q*uD*n*ds - K*grad(q)*n*pD*ds + lambda_p/h*pD*q*ds
    lh += f*q*dx - K*grad(q)*n*pD*ds + lambda_p/h*pD*q*ds

    return fes,invmstar,lh,mh,mesh,dx

def TimeStepping(invmstar, initial_condu = None, initial_condp = None, t0 = 0, tend = 1,
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
        t.Set(time+dt)
        lh.Assemble() 
        res = lh.vec + 1/dt * mh.mat * gfu.vec
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



# Define physical parameters for Biot
E = 1
nu = 0.2
mu  = E/2/(1+nu)
lam = E*nu/(1+nu)/(1-2*nu)
K = 0.1
alpha = 1
M = 100

# Quadrilateral (or simplicial mesh)
quad_mesh = True
# Finite element space order
orderu = 1
orderp = 1
# time step
# dt = 1e-4

# penalty parameters
# lambda_u = 20*lam
# lambda_p = 50*K
# gamma_s = 20*lam
# gamma_p = 10*K

lambda_u = 100
lambda_p = 500*K
# gamma_s = 0
# gamma_p = 0
gamma_s = 20*lam
gamma_p = 10*K
# gamma_m = 0.1/M/dt
gamma_m = 0 


# Manufactured exact solution for monitoring the error
t = Parameter(0.0) # A Parameter is a constant CoefficientFunction the value of which can be changed with the Set-function.
u_x = e**(-t)*sin(pi*x)*sin(pi*y)
u_y = e**(-t)*sin(pi*x)*sin(pi*y)
exact_u = CF((u_x,u_y))
exact_p = e**(-t)*(cos(pi*y)+1)

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
f = (1/M*exact_p+alpha*(u_x.Diff(x)+u_y.Diff(y))).Diff(t) - K*(exact_p.Diff(x).Diff(x)+exact_p.Diff(y).Diff(y)) # source term

# uD = exact_u
# pD = exact_p

# Define level set function
levelset = sqrt(x**2 + y**2) - 0.5



results = []

for k in range(2, 6):
    h0 = 1/2**k
    # h0 = 1e-2
    # h0 = 1/128
    # dt = 1/2**(k+2)
    # dt = 1e-4
    dt = h0**2
    # endT = 1000*dt
    endT = 0.5
    # tau_fpl = h0*h0*alpha*alpha/4/(lam+2*mu)-K*dt+h0*h0/6/M 
    # if tau_fpl < 0:
    #     tau_fpl = 0
    # tau_fpl = 0.1*h0**2
    tau_fpl = 0
    fes,invmstar,lh,mh,mesh,dneg = solve_biot_cutfem(dt, b, f, exact_u, exact_p, levelset, quad_mesh, \
                                            mu,lam,tau_fpl,lambda_u,lambda_p,gamma_s,gamma_p,gamma_m,alpha,M,K, orderu, orderp, h0)
    t.Set(0)
    gfu = GridFunction(fes)
    gfut,gfu = TimeStepping(invmstar, initial_condu=exact_u, initial_condp=exact_p, tend=endT, nsamples=20)
    exact_ = GridFunction(fes)
    t.Set(endT)
    exact_.components[0].Set(exact_u)
    exact_.components[1].Set(exact_p)
    error_u = sqrt(Integrate((gfu.components[0] - exact_u)**2 * dneg, mesh))
    error_p = sqrt(Integrate((gfu.components[1] - exact_p)**2 * dneg, mesh))
    # mask = IfPos(levelset,0,1)
    # error_u = sqrt(Integrate(((gfu.components[0] - exact_u)*mask)**2, mesh))
    # error_p = sqrt(Integrate((mask*(gfu.components[1] - exact_p))**2, mesh))
    ndof = gfu.space.ndof
    results.append((h0,ndof,error_u, error_p ))

print_convergence_table(results)
mask = IfPos(levelset,0,1)
vtk = VTKOutput(mesh,coefs=[mask*(-gfu.components[0]+exact_u),mask*(-gfu.components[1]+exact_p)],names=["uh","ph"],filename="/mnt/d/ngs_output/biot_fem",subdivision=2)
# vtk = VTKOutput(mesh,coefs=[mask, -gfu.components[0]+exact_u,-gfu.components[1]+exact_p],names=["mask","uh","ph"],filename="/mnt/d/ngs_output/biot_fem2",subdivision=2)
vtk.Do()