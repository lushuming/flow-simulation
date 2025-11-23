from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
import numpy as np
import scipy.sparse as sp

ngsglobals.msg_level = 2

def solve_poisson_unfitted(mh, quad_mesh, order, levelset, ifghost, lambda_nitsche, gamma_stab, coeff_f, exactu ):

    # 1. Construct the mesh
    square = SplineGeometry()
    square.AddRectangle((-1, -1), (1, 1), bc=1)
    ngmesh = square.GenerateMesh(maxh=mh, quad_dominated=quad_mesh)
    mesh = Mesh(ngmesh)

    # 2.  Higher order level set approximation
    lsetmeshadap = LevelSetMeshAdaptation(mesh, order=order, threshold=0.1,
                                          discontinuous_qn=True)
    deformation = lsetmeshadap.CalcDeformation(levelset)
    lsetp1 = lsetmeshadap.lset_p1

    # Element, facet and dof marking w.r.t. boundary approximation with lsetp1:
    ci = CutInfo(mesh, lsetp1)
    hasneg = ci.GetElementsOfType(HASNEG)
    hasif = ci.GetElementsOfType(IF)
    # facets used for ghost penalty stabilization:
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif, bnd_val_a=False, bnd_val_b=False) 

    kappaminus = CutRatioGF(ci)
    kappaminus_values = kappaminus.vec.FV().NumPy()

    # 找出非零值中的最小值
    # 这里需要过滤掉那些没有被切割的单元的值（通常是0）
    min_value = 1.0 # 初始化一个比任何比率都大的值
    for value in kappaminus_values:
        if value > 0 and value < min_value:
            min_value = value

    # print(f"所有被切割单元中，比值的最小值为: {min_value}")

    # 3. Construct the unfitted fem space 
    Vhbase = H1(mesh, order=order, dirichlet=[], dgjumps=True)
    Vh = Restrict(Vhbase, hasneg)

    gfu = GridFunction(Vh)
    u, v = Vh.TrialFunction(), Vh.TestFunction()
    h = specialcf.mesh_size
    n = Normalize(grad(lsetp1))

    # integration domains:
    dx = dCut(lsetp1, NEG, definedonelements=hasneg, deformation=deformation) # Omega区域上的积分
    ds = dCut(lsetp1, IF, definedonelements=hasif, deformation=deformation) # 物理区域Omega边界上的积分
    dw = dFacetPatch(definedonelements=ba_facets, deformation=deformation)  # ghost penalty项的积分的patch


    # 4. Construc bilinear form and right hand side 

    a = BilinearForm(Vh, symmetric=False)
    # Diffusion term
    a += grad(u) * grad(v) * dx
    # Nitsche term
    a += -grad(u) * n * v * ds
    a += -grad(v) * n * u * ds
    a += (lambda_nitsche / h) * u * v * ds

    if ifghost: 
        # Ghost penalty stabilization
        if order == 1:
            # a += gamma_stab / h * (u - u.Other()) * (v - v.Other()) * dw
            a += gamma_stab * h * (grad(u) - grad(u.Other()))*n * (grad(v) - grad(v.Other()))*n * dw
        elif order == 2:
            penalty = h * (grad(u) - grad(u.Other()))*n * (grad(v) - grad(v.Other()))*n
            d2un = InnerProduct(n, u.Operator("hesse")*n) - InnerProduct(n, u.Other().Operator("hesse")*n)
            d2vn = InnerProduct(n, v.Operator("hesse")*n) - InnerProduct(n, v.Other().Operator("hesse")*n)
            penalty += h**3* d2un*d2vn
            a += gamma_stab * penalty * dw
        
    
    # R.h.s. term:
    f = LinearForm(Vh)
    f += coeff_f * v * dx
    
    # Assemble system
    a.Assemble()
    f.Assemble()

    # 计算系数矩阵的条件数
    rows,cols,vals = a.mat.COO()
    A = sp.csr_matrix((vals,(rows,cols)))
    conds = np.linalg.cond(A.todense())
        
    # 5. Solve for the free dofs
    gfu.vec.data = a.mat.Inverse(Vh.FreeDofs()) * f.vec
    l2error = sqrt(Integrate((gfu - exactu)**2 * dx, mesh))
    return l2error, gfu.space.ndof, conds, min_value

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'Cond.':>12} | {'MinCut.':>12} | {'L2 Error':>12} | {'Order':>6}")
    print("-" * 70)
    for i, (h, dofs, error, conds, min_value) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{min_value:12.4e} | {error:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_error,_,_ = results[i-1]
            rate = (np.log(prev_error) - np.log(error)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {conds:12.4e} |{min_value:12.4e} | {error:12.4e} | {rate:6.2f}")



# Define important parameters
# Quadrilateral (or simplicial mesh)
quad_mesh = False
# Finite element space order
order = 1
# Stabilization parameter for ghost-penalty
gamma_stab = 1
# Stabilization parameter for Nitsche
lambda_nitsche = 10 * order * order
ifghost = True

r2 = 3 / 4  # outer radius
r1 = 1 / 4  # inner radius
rc = (r1 + r2) / 2.0
rr = (r2 - r1) / 2.0
r = sqrt(x**2 + y**2)
levelset = IfPos(r - rc, r - rc - rr, rc - r - rr) 
# IfPos(cond, val1, val2) 是一个条件函数，如果 cond > 0 返回 val1，否则返回 val2。

exactu = (20 * (r2 - sqrt(x**2 + y**2)) * (sqrt(x**2 + y**2) - r1)).Compile()
coeff_f = - (exactu.Diff(x).Diff(x) + exactu.Diff(y).Diff(y)).Compile()

results = []

for k in range(2, 6):
    mh = 1/2**k
    l2error, ndof, conds, min_value = solve_poisson_unfitted(mh, quad_mesh, order, levelset, ifghost, lambda_nitsche, gamma_stab, coeff_f, exactu )
    results.append((mh,ndof,l2error,conds,min_value))

print_convergence_table(results)
