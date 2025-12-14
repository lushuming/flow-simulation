from netgen.geom2d import SplineGeometry
from ngsolve import *

from xfem import *
from xfem.mlset import *
from math import pi,e
from numpy import linspace
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from scipy.io import savemat

ngsglobals.msg_level = 2

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_linear_elastic_dg_mlsets(h0, quad_mesh, orderu, level_sets,fe, exact_u, mu,lam,beta_u,gamma_u,NitschType):

    # 1. Construct the mesh
    geo = SplineGeometry()
    geo.AddRectangle((0,0), (1, 1), bcs=("outer","outer","outer","left"))
    ngmesh = geo.GenerateMesh(maxh=h0, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    nr_ls = len(level_sets)
    level_sets_p1 = tuple(GridFunction(H1(mesh,order=2)) for i in range(nr_ls))

    for i, lset_p1 in enumerate(level_sets_p1):
        InterpolateToP1(level_sets[i],lset_p1)

    omega = DomainTypeArray((NEG,NEG,NEG,NEG))
    boundary = omega.Boundary()

    mlci = MultiLevelsetCutInfo(mesh,level_sets_p1)
    els_hasneg = mlci.GetElementsWithContribution(omega)
    els_if = mlci.GetElementsWithContribution(boundary)
    facets_gp = GetFacetsWithNeighborTypes(mesh,a=els_hasneg,b=els_if,use_and=True)
    interior_facets = GetFacetsWithNeighborTypes(mesh,a=els_hasneg,b=els_hasneg,bnd_val_a=False,bnd_val_b=False,use_and=True)

    els_if_single = {}
    for i,dtt in enumerate(boundary):
        els_if_single[dtt] = mlci.GetElementsWithContribution(dtt)
        
    # 3. Construct the unfitted DG space 
    # Uhbase = VectorL2(mesh, order=orderu, dirichlet=[], dgjumps=True) # space for displacement
    Uhbase = VectorH1(mesh, order=orderu, dirichlet=[], dgjumps=True) # space for displacement
    U = Compress(Uhbase, GetDofsOfElements(Uhbase, els_hasneg))
    # freedofs = GetDofsOfElements(U,els_hasneg) & U.FreeDofs()
    u,v = U.TnT()
    h = specialcf.mesh_size
    # ne = specialcf.normal(mesh.dim) # normal vectors on faces
    normals = omega.GetOuterNormals(level_sets_p1) # levelsets的外法向量
    domega = dCut(level_sets_p1,omega,definedonelements=els_hasneg)
    # dk = dCut(level_sets_p1, omega, skeleton=True, definedonelements=interior_facets)
    dk = dx(skeleton=True,definedonelements=interior_facets)
    dsc = {dtt:dCut(level_sets_p1,dtt,definedonelements=els_if_single[dtt]) for dtt in boundary}
    dsbar = ds(skeleton=True)
    dw = dFacetPatch(definedonelements=facets_gp)

    # Define special variables
    strain_u = Sym(Grad(u))
    strain_v = Sym(Grad(v))
    # mean_stress_u = 0.5*(Stress(Sym(Grad(u)))+Stress(Sym(Grad(u.Other()))))*ne
    # mean_stress_v = 0.5*(Stress(Sym(Grad(v)))+Stress(Sym(Grad(v.Other()))))*ne
    jump_u = u - u.Other()
    jump_v = v - v.Other()

    # jump_du = Grad(u)*ne - Grad(u.Other())*ne
    # jump_dv = Grad(v)*ne - Grad(v.Other())*ne

    # mask = omega.Indicator(level_sets_p1)
    mask = 1

    # 4. Construc bilinear form and right hand side 
    # stiffness matrix
    Ah = BilinearForm(U)
    Ah += 2*mu*InnerProduct(strain_u,strain_v)*domega + lam*div(u)*div(v)*domega
    # Ah += - mask*(InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u))*dk 
    # boundary terms
    # Ah += - (InnerProduct(Stress(Sym(Grad(u)))*ne,v) + InnerProduct(Stress(Sym(Grad(v)))*ne,u))*dsbar # outer boundary
    # levelset boundaries
    for bnd, n in normals.items():
        Ah += -InnerProduct(Stress(Sym(Grad(u)))*n,v) * dsc[bnd]
        Ah += -InnerProduct(Stress(Sym(Grad(v)))*n,u) * dsc[bnd]
        
    # ghost penalty terms
    Ah += gamma_u / (h**2) * jump_u * jump_v * dw


    # r.h.s
    lh = LinearForm(U) 
    lh += fe*v*domega 
    # lh += - InnerProduct(uD,Stress(Sym(Grad(v)))*ne)*dsbar 
    for bnd, n in normals.items():
        lh += -InnerProduct(uD,Stress(Sym(Grad(v)))*n)*dsc[bnd] 
        
    
    if NitschType == 1:
        # Ah += beta_u/h*InnerProduct(jump_u,jump_v)*dk  # interior jump
        # Ah += beta_u/h*InnerProduct(u,v)*dsbar 
        # lh += beta_u/h*InnerProduct(uD,v)*dsbar
        for bnd, n in normals.items():
            Ah += beta_u / h * InnerProduct(u, v) * dsc[bnd] # nitsche term
            lh += beta_u/h*InnerProduct(uD,v)*dsc[bnd] 
    elif NitschType == 2:
        # Ah += mask*beta_u/h*(2*mu*InnerProduct(jump_u,jump_v)+lam*InnerProduct(jump_u,ne)*InnerProduct(jump_v,ne))*dk  # interior jump
        # Ah += beta_u/h*(2*mu*InnerProduct(u,v) + lam*InnerProduct(u,ne)*InnerProduct(v,ne))*dsbar  
        # lh += beta_u/h*(2*mu*InnerProduct(uD,v) + lam*InnerProduct(uD,ne)*InnerProduct(v,ne))*dsbar
        for bnd, n in normals.items():
            Ah += beta_u/h*(2*mu*InnerProduct(u,v) + lam*InnerProduct(u,n)*InnerProduct(v,n))*dsc[bnd]  
            lh += beta_u/h*(2*mu*InnerProduct(uD,v) + lam*InnerProduct(uD,n)*InnerProduct(v,n))*dsc[bnd]

    Ah.Assemble()
    lh.Assemble()
    gfu = GridFunction(U)
    gfu.vec.data = Ah.mat.Inverse() * lh.vec

    
    # Calculate the condition number of the stiffness matrix
    # rows,cols,vals = Ah.mat.COO()
    # A = sp.csr_matrix((vals,(rows,cols)))
    # conds = np.linalg.cond(A.todense())
    conds = 0
        
    # 5. Calculate L2 error
    error_u = sqrt(Integrate((gfu - exact_u)**2 * domega.order(orderu), mesh))

    gff = GridFunction(U)
    gff.Set(exact_u)

    deltau = CF((exact_u[0].Diff(x),exact_u[0].Diff(y),exact_u[1].Diff(x),exact_u[1].Diff(y)),dims=(2, 2)).Compile()
    grad_error_u = Grad(gfu)-deltau
    error_u_H1 = sqrt(Integrate(InnerProduct(grad_error_u,grad_error_u)*domega.order(orderu), mesh))

    # 计算系数矩阵的条件数
    # rows,cols,vals = Ah.mat.COO()
    # A_scipy = sp.csr_matrix((vals,(rows,cols)))
    # conds = np.linalg.cond(A_scipy.todense())
    # print(conds)
    # 变量名 'K' (Stiffness Matrix) 将在 MATLAB 中使用
    # data_to_save = {'K': A_scipy} 

    # 5. 保存为 .mat 文件
    # output_filename = '/mnt/d/ngsolve_matrix/linearelasticity_' + str(h0) + '_' + str(gamma_u) + '.mat'
    # savemat(output_filename, data_to_save)

    # print(f"矩阵已成功保存到文件: {output_filename}")

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
#(1,200,10)
order_u = 2
beta_u = 200

# parameter of ghost penalty
gamma_u = 0


quad_mesh = False

# manufactured solution
# u_x = 4*(x-1)*(x+1)*(y-0.5)*(y+0.5)
# u_y = 8*(x-1)*(x+1)*(y-0.5)*(y+0.5)

u_x = x**2
u_y = y**2

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
line1 = 1/4 - x
line2 = x - 3/4
line3 = 1/4 - y
line4 = y - 3/4
level_sets = (line1, line2, line3,line4)


results = []
NitschType = 2
for k in range(2, 7):
    h0 = 1/2**k
    error_u,error_u_H1, ndof, conds = solve_linear_elastic_dg_mlsets(h0, quad_mesh, order_u, level_sets,fe, exact_u, mu,lam,beta_u,gamma_u,NitschType)
    results.append((h0,ndof,conds,error_u,error_u_H1))

print_convergence_table(results)
