from netgen.occ import *
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

def solve_biotdarcy_steady_unfitted(mh, order_eta, order_p, levelset, fe, fp, ff, exact_eta, exact_p, exact_f):

    # 1. Construct the mesh
    Omega = Rectangle(1, 1).Face() 
    Omega.faces.name = "Omega"
    Omega.edges.name="outer"
    mesh = Mesh(OCCGeometry(Omega, dim=2).GenerateMesh(maxh=mh))

    # 2. Define the farcture region
    lsetp1 = GridFunction(H1(mesh,order=1))
    InterpolateToP1(levelset,lsetp1)
    ci = CutInfo(mesh,lsetp1)
    haspos = ci.GetElementsOfType(HASPOS)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)

    # Interior faces and ghost faces
    interior_neg_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg)
    interior_pos_facets = GetFacetsWithNeighborTypes(mesh, a=haspos, b=haspos)
    gp_neg_faces = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)
    gp_pos_faces = GetFacetsWithNeighborTypes(mesh, a=haspos, b=hasif)

    # 3. Construct the unfitted fem space 
    Ehbase = VectorL2(mesh, order=order_eta, dirichlet="outer", dgjumps=True) # space for velocity
    Phbase = L2(mesh, order=order_p, dirichlet="outer", dgjumps=True) # space for pressure
    Fhbase = H1(mesh, order=order_p, dirichlet=".*", dgjumps=True) # space for pressure
    E1 = Compress(Ehbase, GetDofsOfElements(Ehbase, hasneg))
    E2 = Compress(Ehbase, GetDofsOfElements(Ehbase, haspos))
    P1 = Compress(Phbase, GetDofsOfElements(Phbase, hasneg))
    P2 = Compress(Phbase, GetDofsOfElements(Phbase, haspos))
    Pf = Compress(Fhbase, GetDofsOfElements(Fhbase, hasif))
    fes = E1*E2*P1*P2*Pf
    (eta1,eta2,p1,p2,pf), (xi1,xi2,q1,q2,qf) = fes.TnT()

    # Define special variables
    h = specialcf.mesh_size
    nf = Normalize(grad(lsetp1)) # normal vector on the fracture
    ne = specialcf.normal(2) # normal vector on edges
    
    # Define the jumps and the averages
    jump_eta1 = eta1 - eta1.Other()
    jump_eta2 = eta2 - eta2.Other()
    jump_xi1 = xi1 - xi1.Other()
    jump_xi2 = xi2 - xi2.Other()
    jump_p1 = p1 - p1.Other()
    jump_p2 = p2 - p2.Other()
    jump_q1 = q1 - q1.Other()
    jump_q2 = q2 - q2.Other()
    jump_pf = pf - pf.Other()
    jump_qf = qf - qf.Other()
    
    strain_eta1 = Sym(Grad(eta1))
    strain_eta2 = Sym(Grad(eta2))
    strain_xi1 = Sym(Grad(xi1))
    strain_xi2 = Sym(Grad(xi2))
    mean_stress_eta1 = 0.5*(Stress(Sym(Grad(eta1)))+Stress(Sym(Grad(eta1.Other()))))*ne
    mean_stress_eta2 = 0.5*(Stress(Sym(Grad(eta2)))+Stress(Sym(Grad(eta2.Other()))))*ne
    mean_stress_xi1 = 0.5*(Stress(Sym(Grad(xi1)))+Stress(Sym(Grad(xi1.Other()))))*ne
    mean_stress_xi2 = 0.5*(Stress(Sym(Grad(xi2)))+Stress(Sym(Grad(xi2.Other()))))*ne
    
    mean_dp1dn = 0.5*Kp*(grad(p1)+grad(p1.Other()))*ne
    mean_dq1dn = 0.5*Kp*(grad(q1)+grad(q1.Other()))*ne
    mean_dp2dn = 0.5*Kp*(grad(p2)+grad(p2.Other()))*ne
    mean_dq2dn = 0.5*Kp*(grad(q2)+grad(q2.Other()))*ne
    mean_dpfdn = 0.5*Kft*(grad(pf)+grad(pf.Other()))*ne
    mean_dqfdn = 0.5*Kft*(grad(qf)+grad(qf.Other()))*ne
    
    mean_p1 = 0.5*(p1 + p1.Other())
    mean_q1 = 0.5*(q1 + q1.Other())
    mean_p2 = 0.5*(p2 + p2.Other())
    mean_q2 = 0.5*(q2 + q2.Other())
 

    # integration operators
    # Element-wise integrals
    dx_neg = dCut(lsetp1, NEG, definedonelements=hasneg)
    dx_pos = dCut(lsetp1, POS, definedonelements=haspos)
    dgamma = dCut(lsetp1, IF, definedonelements=hasif)
    
    # Interior skeleton integrals:
    dk_neg = dCut(lsetp1, NEG, skeleton=True, definedonelements=interior_neg_facets)
    dk_pos = dCut(lsetp1, POS, skeleton=True, definedonelements=interior_pos_facets)
    
    # Domain boundary integrals
    dso_neg = ds(skeleton=True)
    dso_pos = ds(skeleton=True)
    
    # Ghost penalty integrals
    dw_neg = dFacetPatch(definedonelements=gp_neg_faces)
    dw_pos = dFacetPatch(definedonelements=gp_pos_faces)



    # 4. Construc bilinear form and right hand side 

    ah = BilinearForm(fes)
    ####################### Equation 1 ###################
    # Am
    ah += 2*mu*InnerProduct(strain_eta1,strain_xi1)*dx_neg + lam*div(eta1)*div(xi1)*dx_neg \
            - (InnerProduct(mean_stress_eta1,jump_xi1) + InnerProduct(mean_stress_xi1,jump_eta1) - beta_eta/h*InnerProduct(jump_eta1,jump_xi1))*dk_neg \
            - (InnerProduct(Stress(Sym(Grad(eta1)))*ne,xi1) + InnerProduct(Stress(Sym(Grad(xi1)))*ne,eta1) - beta_eta/h*InnerProduct(eta1,xi1))*dso_neg
    ah += 2*mu*InnerProduct(strain_eta2,strain_xi2)*dx_pos + lam*div(eta2)*div(xi2)*dx_pos \
            - (InnerProduct(mean_stress_eta2,jump_xi2) + InnerProduct(mean_stress_xi2,jump_eta2) - beta_eta/h*InnerProduct(jump_eta2,jump_xi2))*dk_pos \
            - (InnerProduct(Stress(Sym(Grad(eta2)))*ne,xi2) + InnerProduct(Stress(Sym(Grad(xi2)))*ne,eta2) - beta_eta/h*InnerProduct(eta2,xi2))*dso_pos
    
    # Bm
    ah += -alpha*(div(xi1)*p1*dx_neg - mean_p1*jump_xi1*ne*dk_neg - p1*xi1*ne*dso_neg)
    ah += -alpha*(div(xi2)*p2*dx_pos - mean_p2*jump_xi2*ne*dk_pos - p2*xi2*ne*dso_pos)
    
    # I
    ah += -alpha*pf*nf*(xi1-xi2)*dgamma
    
    # ghost penalty for eta
    ah += sigma_eta / (h**2) * (eta1 - eta1.Other()) * (xi1 - xi1.Other()) * dw_neg
    ah += sigma_eta / (h**2) * (eta2 - eta2.Other()) * (xi2 - xi2.Other()) * dw_pos
    
    ####################### Equation 2 ###################
    
    # Ap
    ah += Kp*grad(p1)*grad(q1)*dx_neg \
            - (mean_dp1dn*jump_q1 + mean_dq1dn*jump_p1 - beta_p/h*jump_p1*jump_q1)*dk_neg \
            - (Kp*grad(p1)*ne*q1 + Kp*grad(q1)*ne*p1 - beta_p/h*p1*q1)*dso_neg
    ah += Kp*grad(p2)*grad(q2)*dx_pos \
            - (mean_dp2dn*jump_q2 + mean_dq2dn*jump_p2 - beta_p/h*jump_p2*jump_q2)*dk_pos \
            - (Kp*grad(p2)*ne*q2 + Kp*grad(q2)*ne*p2 - beta_p/h*p2*q2)*dso_pos
    # I 
    ah += (alphaf *(0.5*(p1+p2) - pf)*0.5*(q1+q2) + betaf*(p1-p2)*(q1-q2))*dgamma
    
    # ghost penalty for pressure
    ah += sigma_p / (h**2) * (p1 - p1.Other()) * (q1 - q1.Other()) * dw_neg
    ah += sigma_p / (h**2) * (p2 - p2.Other()) * (q2 - q2.Other()) * dw_pos
    
    ####################### Equation 3 ###################
    # Af
    ah += Kft*grad(pf)*grad(qf)*dgamma 
    
    # I
    ah += (-alphaf *(0.5*(p1+p2) - pf)*qf + betaf*(p1-p2)*(q1-q2))*dgamma
    
    
    ah.Assemble()

    # r.h.s
    fh = LinearForm(fes)
    ####################### Equation 1 ###################
    fh += fe*xi1*dx_neg - InnerProduct(etaD,Stress(Sym(Grad(xi1)))*ne)*dso_neg + beta_eta/h*etaD*xi1*dso_neg
    fh += fe*xi2*dx_pos - InnerProduct(etaD,Stress(Sym(Grad(xi2)))*ne)*dso_pos + beta_eta/h*etaD*xi2*dso_pos
    
    ####################### Equation 2 ###################
    fh += fp*q1*dx_neg - Kp*grad(q1)*ne*pD*dso_neg + beta_p/h*pD*q1*dso_neg
    fh += fp*q2*dx_pos - Kp*grad(q2)*ne*pD*dso_pos + beta_p/h*pD*q2*dso_pos
    
    ####################### Equation 3 ###################
    fh += ff*qf*dgamma
    
    fh.Assemble()

        
    # 5. Solve for the free dofs
    gfu = GridFunction(fes)
    freedofs = fes.FreeDofs()
    gfu.components[4].Set(pfD,BND)
    fh.vec.data -= ah.mat * gfu.vec
    gfu.vec.data += ah.mat.Inverse(freedofs) * fh.vec
    
    error_eta = sqrt(Integrate((gfu.components[0] - exact_eta)**2 * dx_neg + (gfu.components[1] - exact_eta)**2 * dx_pos, mesh))
    error_p = sqrt(Integrate((gfu.components[2] - exact_p)**2 * dx_neg + (gfu.components[3] - exact_p)**2 * dx_pos, mesh))
    error_f = sqrt(Integrate((gfu.components[4]-exact_f)**2*dgamma, mesh))
    

    return error_eta, error_p, error_f, gfu.space.ndof

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'Error_eta.':>12}  | {'Order':>6} | {'Error_p.':>12}  | {'Order':>6} | {'Error_f':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h,ndof,error_eta, error_p, error_f) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {ndof:8d} | {error_eta:12.4e}  | {'-':>6} |{error_p:12.4e} | {'-':>6}| {error_f:12.4e} | {'-':>6}")
        else:
            prev_h,_,prev_error_eta, prev_error_p,prev_error_f = results[i-1]
            rate_eta = (np.log(prev_error_eta) - np.log(error_eta)) / (np.log(prev_h) - np.log(h))
            rate_p = (np.log(prev_error_p) - np.log(error_p)) / (np.log(prev_h) - np.log(h))
            rate_f = (np.log(prev_error_f) - np.log(error_f)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {ndof:8d} | {error_eta:12.4e} | {rate_eta:6.2f} |{error_p:12.4e}| {rate_p:6.2f} | {error_f:12.4e} | {rate_f:6.2f}")



# Define important parameters
# P1 (10,10,0.1,0.1)
# p2 收敛阶不对
order_eta = 2
order_p = 2
# Nitsche penalty
beta_eta = 50
beta_p = 50
# ghost penalty
sigma_eta = 0.01
sigma_p = 0.005
# physical parameters
mu  = 1
lam = 1e-5
alpha = 1e-14
Kp = 1
Kft = 100
Kfn = 100
d = 1e-4

ksi = 0.75
alphaf = 4*Kfn/d/(2*ksi-1)
betaf = Kfn/d



    
# 定义解析解
# eta_x = 1 - 2*y
# eta_y = 1 + 2*x
eta_x = sin(pi*y) + y
eta_y = (y-1/2)**2 - x
exact_eta = CF((eta_x,eta_y))
exact_p = IfPos(y-1/2, y*sin(pi*x), (y-1/betaf)*sin(pi*x))
exact_f = (1-1/betaf)/2*sin(pi*x)
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

# 向量形式
fe = CF( (f_x,f_y) )
fp = -Kp*(exact_p.Diff(x).Diff(x)+exact_p.Diff(y).Diff(y))
ff = -Kft*(exact_f.Diff(x).Diff(x)+exact_f.Diff(y).Diff(y))

etaD = exact_eta
pD = exact_p
pfD = exact_f


# Set level set function
levelset = y-1/2

results = []

for k in range(2, 7):
    mh = 1/2**k
    error_eta, error_p, error_f, ndof = solve_biotdarcy_steady_unfitted(mh, order_eta, order_p, levelset, fe, fp, ff, exact_eta, exact_p, exact_f)
    results.append((mh,ndof,error_eta, error_p, error_f))

print_convergence_table(results)
