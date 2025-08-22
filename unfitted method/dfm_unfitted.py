from math import pi 
from ngsolve import *
from xfem import *
from netgen.occ import *
from ngsolve.webgui import *
import numpy as np

def solve_dfm_unfitted(f, gD, order=1, mh=0.1):
    """
    解决带有非齐次Dirichlet和Neumann边界条件的elliptic问题:
        -Δu + alpha*u = f  在域内
        u = gD   在 dirichlet_bdr 上
        ∂u/∂n = gN 在 neumann_bdr 上（可选）

    参数：
        f              : CoefficientFunction 或可表达右端项的表达式
        gD             : CoefficientFunction 或表达Dirichlet边界条件的表达式
        dirichlet_bdr  : 字符串,指定Dirichlet边界的名称,比如 "left|bottom"
        gN             : CoefficientFunction 或表达Neumann边界条件的表达式,可选
        neumann_bdr    : 字符串,指定Neumann边界的名称,可选
        order          : 有限元阶数,默认2
        mesh           : Mesh 对象,可选,不传则默认用unit_square生成
        h              : mesh size

    返回：
        求解得到的GridFunction u
    """

    # 1. Construct the mesh
    geo = OCCGeometry(unit_square_shape.Scale((0,0,0),1), dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=mh,quad_dominated=False))

    # 2. Define the levelset function for the interface
    levelset = x-1/2
    lsetp1 = GridFunction(H1(mesh,order=order))
    InterpolateToP1(levelset,lsetp1)
    DrawDC(lsetp1,-1,1,mesh,'lsetp1')

    # Construct the unfitted fem space 
    Vhbase = H1(mesh,order=1,dirichlet='.*',dgjumps=True)
    ci = CutInfo(mesh,lsetp1)
    Vh = FESpace([Compress(Vhbase,GetDofsOfElements(Vhbase,ci.GetElementsOfType(cdt))) for cdt in [HASNEG, HASPOS,IF]])

    # 3. Construct the bilinear forms and the right hand side

    # 3.1 Define averages and jumps
    u,v = Vh.TnT()
    h = specialcf.mesh_size
    n = specialcf.normal(2)
    n_gamma = Normalize(grad(lsetp1))
    jump_grad_u0 = (grad(u[0]) - grad(u[0].Other()))*n
    jump_grad_v0 = (grad(v[0]) - grad(v[0].Other()))*n
    jump_grad_u1 = (grad(u[1]) - grad(u[1].Other()))*n
    jump_grad_v1 = (grad(v[1]) - grad(v[1].Other()))*n
    jump_grad_u2 = (grad(u[2]) - grad(u[2].Other()))*n
    jump_grad_v2 = (grad(v[2]) - grad(v[2].Other()))*n
    jump_u = (grad(u[0]) - grad(u[1]))*n_gamma
    jump_v = (grad(v[0]) - grad(v[1]))*n_gamma

    partialOmega1 = x*(x-1/2)*y*(1-y)
    pO1 = GridFunction(H1(mesh,order=1))
    InterpolateToP1(partialOmega1,pO1)
    c1 = CutInfo(mesh, pO1)
    boundaryEle1 = c1.GetElementsOfType(IF)
    
    partialOmega2 = (1-x)*(1/2-x)*y*(1-y)
    pO2 = GridFunction(H1(mesh,order=1))
    InterpolateToP1(partialOmega2,pO2)
    c2 = CutInfo(mesh, pO2)
    boundaryEle2 = c2.GetElementsOfType(IF)

    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    haspos = ci.GetElementsOfType(HASPOS)
    hasif = ci.GetElementsOfType(IF)
    
    # # facets used for stabilization:
    fh1_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=boundaryEle1,bnd_val_a=False,bnd_val_b=False,use_and=True)
    fh2_facets = GetFacetsWithNeighborTypes(mesh, a=haspos, b=boundaryEle2,bnd_val_a=False,bnd_val_b=False,use_and=True)
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasif, b=hasif, bnd_val_a=False,bnd_val_b=False,use_and=True) # all interior faces in T_{h,Gamma}

    # Construc bilinear form and right hand side 
    
    ## integral operators
    dx_neg = dCut(levelset=lsetp1, domain_type = NEG, definedonelements=ci.GetElementsOfType(HASNEG))
    dx_pos = dCut(levelset=lsetp1, domain_type = POS, definedonelements=ci.GetElementsOfType(HASPOS))
    ds = dCut(levelset=lsetp1, domain_type = IF, definedonelements=ci.GetElementsOfType(IF))
    df0 = dFacetPatch(definedonelements=fh1_facets)
    df1 = dFacetPatch(definedonelements=fh2_facets)
    df2 = dFacetPatch(definedonelements=ba_facets)
    ## Bilinear form
    ah = BilinearForm(Vh,symmetric=True)
    ah += grad(u[0]) * grad(v[0]) * dx_neg +  grad(u[1]) * grad(v[1]) * dx_pos + grad(u[2]) * grad(v[2]) * ds
    ah += ((u[0]-u[2]) * (v[0]-v[2]) + (u[1]-u[2]) * (v[1]-v[2]) )* ds
    # stabilization terms
    # ah += h * jump_grad_u0 * jump_grad_v0 * df0
    # ah += h * jump_grad_u1 * jump_grad_v1 * df1
    # ah += (h * jump_grad_u2 * jump_grad_v2 * df2 + h * jump_u * jump_v * ds)
    # stabilization terms
    # ah += h * jump_grad_u0 * jump_grad_v0 * dx(skeleton=True,definedonelements=fh1_facets)
    # ah += h * jump_grad_u1 * jump_grad_v1 * dx(skeleton=True,definedonelements=fh2_facets)
    # ah += h * jump_grad_u0 * jump_grad_v0 * dx(skeleton=True,definedonelements=ba_facets)
    # ah += h * jump_grad_u1 * jump_grad_v1 * dx(skeleton=True,definedonelements=ba_facets)
    ah += h * jump_grad_u2 * jump_grad_v2 * dx(skeleton=True,definedonelements=ba_facets) 
    ah += h * jump_u * jump_v * ds

    ah. Assemble()
    
    ## right hand side
    F = LinearForm(Vh)
    F += ( f[0] * v[0] * dx_neg + f[1] * v[1] * dx_pos + f[2] * v[2] * ds)
    F.Assemble()

    # 5. Solve for the free dofs
    # Solve
    gfu = GridFunction(Vh)
    freedofs = Vh.FreeDofs()
    ## deal with Dirichlet boundary
    gfu.components[0].Set(gD,BND)
    gfu.components[1].Set(gD,BND)
    F.vec.data -= ah.mat * gfu.vec
    gfu.vec.data += ah.mat.Inverse(freedofs) * F.vec
    #print(f"gfu.vec.size = {gfu.vec.size}")

    gfu_orig = GridFunction(Vhbase)
    gfu_orig.vec[:] = 0  # 默认设置为 0
    used_dofs = GetDofsOfElements(Vhbase,ci.GetElementsOfType(HASNEG))
    # 将压缩空间上的向量嵌入到原始空间
    dof_indices = [i for i in range(len(used_dofs)) if used_dofs[i]]
    for i, dof in enumerate(dof_indices):
        gfu_orig.vec[dof] += gfu.components[0].vec[i]
    
    used_dofs = GetDofsOfElements(Vhbase,ci.GetElementsOfType(HASPOS))
    # 将压缩空间上的向量嵌入到原始空间
    dof_indices = [i for i in range(len(used_dofs)) if used_dofs[i]]
    for i, dof in enumerate(dof_indices):
        gfu_orig.vec[dof] += gfu.components[1].vec[i]
    
    used_dofs = GetDofsOfElements(Vhbase,ci.GetElementsOfType(IF))
    # 将压缩空间上的向量嵌入到原始空间
    dof_indices = [i for i in range(len(used_dofs)) if used_dofs[i]]
    for i, dof in enumerate(dof_indices):
        gfu_orig.vec[dof] -= gfu.components[2].vec[i]

    return gfu_orig,mesh

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'L2 Error':>12} | {'Order':>6}")
    print("-" * 45)
    for i, (h, dofs, error) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {error:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_error = results[i-1]
            rate = (np.log(prev_error) - np.log(error)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {error:12.4e} | {rate:6.2f}")



# Define important parameters
order = 1
f = [2*x*(1-x)+2*y*(1-y), 2*x*(1-x)+2*y*(1-y),1/2]
gD = 0
exact_u_p_neg = x*(1-x)*y*(1-y)
exact_u_p_pos = x*(1-x)*y*(1-y)
exact_u_f = y*(1-y)/4
results = []

for k in range(1, 5):
    mh = 1/2**k
    gfu,mesh = solve_dfm_unfitted(f, gD, order, mh)
    # L2误差计算
    err = sqrt(Integrate((gfu - exact_u_p_neg)**2*dx, mesh)) 
    results.append((mh,gfu.space.ndof,err))

print_convergence_table(results)