from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
from math import pi,e
from numpy import linspace
import numpy as np
import scipy.sparse as sp
from scipy.io import savemat

ngsglobals.msg_level = 2

def Stress(strain):
    return 2*mu*strain + lam*Trace(strain)*Id(2)

def solve_linear_elastic_unfitted(h0, t0, quad_mesh, orderu, levelset,fe, exact_u, mu,lam,beta_u,gamma_u,NitschType):

    # 1. Construct the mesh
    square = SplineGeometry()
    # square.AddRectangle((-1, -1), (1, 1), bc=1)
    square.AddRectangle((-1.5, -1.5), (1.5, 1.5), bc=1)
    ngmesh = square.GenerateMesh(maxh=h0, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=1, threshold=0.2,
                                      discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1

    # lsetp1 = GridFunction(H1(mesh,order=orderu))
    # InterpolateToP1(levelset,lsetp1)

    
    # Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    # facets used for ghost penalty stabilization:
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif)
    interior_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasneg)

    

    ######### Add a threshhold to decide whether small cut occurs ####
    # kappaminus = CutRatioGF(ci)
    # kappaminus_values = kappaminus.vec.FV().NumPy()
    # positive_values = [v for v in kappaminus_values if v > 0]
    # min_value_pythonic = min(positive_values)
    # if min_value_pythonic > t0*h0*h0/2:
    #     gamma_u = 0
    #     print(f"There are no small cuts.")
    # else:
    #     print("There are small cuts.")

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
    df = dx(skeleton=True,definedonelements=ba_facets,deformation=deformation)
    dw = dFacetPatch(definedonelements=ba_facets, deformation=deformation)

    # 4. Construc bilinear form and right hand side 
    # main parts of Ah and lh
    # stiffness matrix
    Ah = BilinearForm(U)
    Ah += 2*mu*InnerProduct(strain_u,strain_v)*domega + lam*div(u)*div(v)*domega \
            - (InnerProduct(mean_stress_u,jump_v) + InnerProduct(mean_stress_v,jump_u))*dk \
            - (InnerProduct(Stress(Sym(Grad(u)))*n,v) + InnerProduct(Stress(Sym(Grad(v)))*n,u))*ds
    # r.h.s
    lh = LinearForm(U) 
    lh += fe*v*domega - InnerProduct(uD,Stress(Sym(Grad(v)))*n)*ds 
    
    ################################# Nitsche penalty terms #################################
    if NitschType == 1:
        Ah += beta_u/h*InnerProduct(jump_u,jump_v)*dk  # interior jump
        Ah += beta_u/h*InnerProduct(u,v)*ds 
        lh += beta_u/h*InnerProduct(uD,v)*ds
    elif NitschType == 2:
        Ah += beta_u/h*(2*mu*InnerProduct(jump_u,jump_v)+lam*InnerProduct(jump_u,ne)*InnerProduct(jump_v,ne))*dk  # interior jump
        Ah += beta_u/h*(2*mu*InnerProduct(u,v) + lam*InnerProduct(u,n)*InnerProduct(v,n))*ds 
        lh += beta_u/h*(2*mu*InnerProduct(uD,v) + lam*InnerProduct(uD,n)*InnerProduct(v,n))*ds
    elif NitschType == 3:
        Ah += beta_u/h*InnerProduct(jump_u,jump_v)*dk  # interior jump
        Ah += beta_u2*h*(div(u)-div(u.Other()))*(div(v)-div(v.Other()))*dk
        Ah += beta_u/h*InnerProduct(u,v)*ds 
        lh += beta_u/h*InnerProduct(uD,v)*ds
    elif NitschType == 4:
        Ah += beta_u/h*(2*mu*InnerProduct(jump_u,jump_v)+lam*InnerProduct(jump_u,ne)*InnerProduct(jump_v,ne))*dk  # interior jump
        Ah += beta_u/h*(2*mu*InnerProduct(u,v) + lam*InnerProduct(u,n)*InnerProduct(v,n))*ds # boundary nitsche term
        Ah += beta_u2/h*InnerProduct(Sym(Grad(u)),Sym(Grad(v)))*ds
        lh += beta_u/h*(2*mu*InnerProduct(uD,v) + lam*InnerProduct(uD,n)*InnerProduct(v,n))*ds
        uB = GridFunction(U)
        uB.Set(exact_u)
        lh += beta_u2/h*InnerProduct(Sym(Grad(uB)),Sym(Grad(v)))*ds


    ################################# ghost penalty terms #################################
    # order=1 i_s 
    Ah += gamma_u * h * ((Grad(u) - Grad(u.Other()))*ne) * ((Grad(v) - Grad(v.Other()))*ne) * df
    # Ah += gamma_u * h * InnerProduct(Grad(u) - Grad(u.Other()),Grad(v) - Grad(v.Other())) * dw
    # Ah += gamma_u  * InnerProduct(Grad(u) - Grad(u.Other()),Grad(v) - Grad(v.Other())) * dw
    # Ah += gamma_u * h * (jump_du*ne) * (jump_dv * ne) * dw
    # Ah += gamma_u / (h**2) * jump_u * jump_v * dw
    # Ah += gamma_u * InnerProduct(Sym(Grad(jump_u)),Sym(Grad(jump_v)))*dw

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
    error_u = sqrt(Integrate((gfu - exact_u)**2 * domega, mesh))

    gff = GridFunction(U)
    gff.Set(exact_u)

    #### 目前能量范数计算不正确
    # err = gfu - gff
    # strain_error = 0.5 * (Grad(err) + Grad(err).trans)
    # # div_error = div(gfu) - div(gff)
    # energy_error_sq = 0
    # if NitschType == 1:
    #     energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    #     energy_error_sq += Integrate(h*err*err*ds,mesh)
    # elif NitschType == 2:
    #     grad_error_u = Grad(err)
    #     # energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    #     # energy_error_sq += Integrate(InnerProduct(grad_error_u,grad_error_u)*  domega, mesh) # gradient error
    #     energy_error_sq += Integrate(h*InnerProduct(Stress(strain_error), Stress(strain_error))*ds,mesh) 
    #     # energy_error_sq += Integrate(h*err*err*ds,mesh)
    #     # energy_error_sq += Integrate(1/h*(2*mu*err*err+lam*(err*n)**2)*ds,mesh)
    # elif NitschType == 3:
    #     energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    #     energy_error_sq += Integrate(h*err*err*ds,mesh)
    # elif NitschType == 4:
    #     energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    #     energy_error_sq += Integrate(h*err*err*ds,mesh)

    # energy_error_sq += Integrate(InnerProduct(strain_error,strain_error)*domega,mesh) + Integrate(beta_u/h*InnerProduct(gfu - gff,gfu - gff)*ds,mesh)
    # energy_error_sq += Integrate(lam* div_error*div_error*domega,mesh)
    # energy_error_sq += Integrate(InnerProduct(strain_error, strain_error)*domega,mesh)
    # energy_error_sq += Integrate(1/h*(2*mu*InnerProduct(gfu - gff,gfu - gff) + lam*((gfu - gff)*n)*((gfu - gff)*n))*ds ,mesh)
    # energy_error_sq += gamma_u  * InnerProduct(Grad(err) - Grad(gfu.Other()-gff.Other),Grad(err) - Grad(err.Other())) * dw
    # error_u_H1 = sqrt(energy_error_sq)
    

    deltau = CF((exact_u[0].Diff(x),exact_u[0].Diff(y),exact_u[1].Diff(x),exact_u[1].Diff(y)),dims=(2, 2)).Compile()
    grad_error_u = Grad(gfu)-deltau
    error_u_H1 = sqrt(Integrate(InnerProduct(grad_error_u,grad_error_u)*domega, mesh))

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
# P1, (Nitsche type 3, 10, 1e-5*lam*lam, 10),(Nitsche type 2, 10, 0, 0.1)
# P2, (Nitsche type 3, 60, 1e-5*lam*lam, )
# P2, (Nitsche type 1, 1000, 20),(Nitsche type 2, 400, 20)
nitschType = 2
order_u = 2
beta_u = 100
beta_u2 = 0

# parameter of ghost penalty
gamma_u = 0.1
t0 = 100 # threshold for small elements

quad_mesh = False


########## divergence free example ##########
# u_x = sin(pi*x) * cos(pi*y) 
# u_y = -cos(pi*x) * sin(pi*y) 

########## Example 1 ##########
u_x = sin(pi*x)*sin(pi*y)
u_y = x*y*(x-1)*(y-1)

########## Example 2 ##########
# u_x = -x*x*y*(2*y-1)*(x-1)*(x-1)*(y-1)
# u_y = x*y*y*(2*x-1)*(x-1)*(y-1)*(y-1)

########## Example 3 ##########
# u_x = sin(pi*x)*sin(pi*y) + x/lam
# u_y = cos(pi*x)*cos(pi*y) + y/lam


# manufactured solution
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
# levelset = sqrt(x**2 + y**2) - 1/2
# levelset = x - 1/3
#### 心型线 ####
levelset = (x**2+y**2-1)**3 - x**2*y**3

results = []

for k in range(2, 6):
    h0 = 1/2**k
    error_u,error_u_H1, ndof, conds = solve_linear_elastic_unfitted(h0, t0, quad_mesh, order_u, levelset,fe, exact_u, mu,lam,beta_u,gamma_u,nitschType)
    results.append((h0,ndof,conds,error_u,error_u_H1))

print_convergence_table(results)
