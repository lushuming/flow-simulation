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

def solve_linear_elastic_unfitted(h0, quad_mesh, orderu, levelset,fe, exact_u, mu,lam,beta_u,gamma_u,NitschType):

    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=orderu, threshold=0.1,
                                      discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1
    # InterpolateToP1(levelset,lsetp1)

    # Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    # facets used for ghost penalty stabilization:
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)
    interior_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg)

    # 3. Construct the unfitted DG space 
    Uhbase = VectorL2(mesh, order=order_u, dirichlet=[], dgjumps=True) # space for displacement
    U = Compress(Uhbase, GetDofsOfElements(Uhbase, ci.GetElementsOfType(HASNEG)))
    u,v = U.TnT()

    # Define special variables
    n = Normalize(grad(lsetp1)) # outer normal vector on the boundary
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
    domega = dCut(lsetp1, NEG, definedonelements=hasneg, deformation=deformation)
    dk = dCut(lsetp1, NEG, skeleton=True, definedonelements=interior_facets,
            deformation=deformation)
    ds = dCut(lsetp1, IF, definedonelements=hasif, deformation=deformation)
    # dw = dCut(lsetp1, NEG, skeleton=True, definedonelements=ba_facets,
    #           deformation=deformation)
    # dw = dx(skeleton=True,definedonelements=ba_facets,deformation=deformation)
    dw = dFacetPatch(definedonelements=ba_facets, deformation=deformation)

    # 4. Construc bilinear form and right hand side 

    Ah = BilinearForm(U)
    # Ae
    # Ah += 2*mu*InnerProduct(strain_u,strain_v)*domega + lam*div(u)*div(v)*domega \
    #         - (InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u) - beta_u/h*InnerProduct(jump_u,jump_v))*dk \
    #         - (InnerProduct(Stress(Sym(Grad(u)))*n,v) + InnerProduct(Stress(Sym(Grad(v)))*n,u) - beta_u/h*InnerProduct(u,v))*ds
    Ah += 2*mu*InnerProduct(strain_u,strain_v)*domega + lam*div(u)*div(v)*domega \
            - (InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u))*dk \
            - (InnerProduct(Stress(Sym(Grad(u)))*n,v) + InnerProduct(Stress(Sym(Grad(v)))*n,u))*ds
    # Nitsch term
    if NitschType == 1:
        Ah += beta_u/h*InnerProduct(jump_u,jump_v)*dk + beta_u/h*InnerProduct(u,v)*ds
    elif NitschType == 2:
        Ah += beta_u/h*(2*mu*InnerProduct(jump_u,jump_v)+lam*(jump_u*ne)*(jump_v*ne))*dk  # interior jump
        Ah += beta_u/h*(2*mu*InnerProduct(u,v) + lam*(u*n)*(v*n))*ds 
    elif NitschType == 3:
        Ah += beta_u/h*(InnerProduct(jump_u,jump_v)+lam*lam*(div(u)-div(u.Other()))*(div(v)-div(v.Other())))*dk  # interior jump
        Ah += beta_u/h*InnerProduct(u,v)*ds

    
    # order=1 i_s 
    # Ah += gamma_u * h * ((Grad(u) - Grad(u.Other()))*ne) * ((Grad(v) - Grad(v.Other()))*ne) * dw
    # Ah += gamma_u * h * InnerProduct(Grad(u) - Grad(u.Other()),Grad(v) - Grad(v.Other())) * dw
    # Ah += gamma_u  * InnerProduct(Grad(u) - Grad(u.Other()),Grad(v) - Grad(v.Other())) * dw
    # Ah += gamma_u * h * (jump_du*ne) * (jump_dv * ne) * dw
    Ah += gamma_u / (h**2) * jump_u * jump_v * dw

    Ah.Assemble()

    # r.h.s
    lh = LinearForm(U) 
    lh += fe*v*domega - InnerProduct(uD,Stress(Sym(Grad(v)))*n)*ds 
    if NitschType == 1:
        lh += beta_u/h*uD*v*ds
    elif NitschType == 2:
        lh += beta_u/h*(2*mu*InnerProduct(uD,v) + lam*(uD*n)*(v*n))*ds
    elif NitschType == 3:
        lh += beta_u/h*uD*v*ds


    lh.Assemble()

    gfu = GridFunction(U)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec

    
    # Calculate the condition number of the stiffness matrix
    rows,cols,vals = Ah.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    # conds = np.linalg.cond(A.todense())
    conds = 0
        
    # 5. Calculate L2 error
    error_u = sqrt(Integrate((gfu - exact_u)**2 * domega, mesh))

    gff = GridFunction(U)
    gff.Set(exact_u)

    #### 目前能量范数计算不正确
    err = gfu - gff
    strain_error = 0.5 * (Grad(err) + Grad(err).trans)
    # div_error = div(gfu) - div(gff)
    energy_error_sq = 0
    if NitschType == 1:
        energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
        energy_error_sq += Integrate(beta_u/h*err*err*ds,mesh)
    elif NitschType == 2:
        grad_error_u = Grad(err)
        energy_error_sq += Integrate(InnerProduct(grad_error_u,grad_error_u)*  domega, mesh) # gradient error
        # energy_error_sq += Integrate(h*InnerProduct(Stress(strain_error), Stress(strain_error))*ds,mesh) 
        energy_error_sq += Integrate(1/h*(2*mu*err*err+lam*(err*n)**2)*ds,mesh)

    # energy_error_sq += Integrate(InnerProduct(strain_error,strain_error)*domega,mesh) + Integrate(beta_u/h*InnerProduct(gfu - gff,gfu - gff)*ds,mesh)
    # energy_error_sq += Integrate(lam* div_error*div_error*domega,mesh)
    # energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    # energy_error_sq += Integrate(1/h*(2*mu*InnerProduct(gfu - gff,gfu - gff) + lam*((gfu - gff)*n)*((gfu - gff)*n))*ds ,mesh)
    # energy_error_sq += gamma_u  * InnerProduct(Grad(err) - Grad(gfu.Other()-gff.Other),Grad(err) - Grad(err.Other())) * dw
    error_u_H1 = sqrt(energy_error_sq)
    

    # err = gfu - gff
    # error_u_H1 = sqrt(Integrate(InnerProduct(Stress(Sym(Grad(err))),Stress(Sym(Grad(err))))* domega, mesh))
    # grad_error_u = Grad(gfu - gff)
    # error_u_H1 = sqrt(Integrate(InnerProduct(grad_error_u,grad_error_u)*  domega, mesh))

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
# P1, 500, 50
# P2, (Nitsche type 1, 1000, 20),(Nitsche type 2, 400, 20)
nitschType = 1
order_u = 2
beta_u = 1000

# parameter of ghost penalty
gamma_u = 10

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
    error_u,error_u_H1, ndof, conds = solve_linear_elastic_unfitted(h0, quad_mesh, order_u, levelset,fe, exact_u, mu,lam,beta_u,gamma_u,nitschType)
    results.append((h0,ndof,conds,error_u,error_u_H1))

print_convergence_table(results)
