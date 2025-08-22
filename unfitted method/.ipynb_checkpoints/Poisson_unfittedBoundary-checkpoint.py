from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *

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
    ba_facets = GetFacetsWithNeighborTypes(mesh, a=hasneg, b=hasif) 

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
        a += gamma_stab / h * (u - u.Other()) * (v - v.Other()) * dw
    
    # R.h.s. term:
    f = LinearForm(Vh)
    f += coeff_f * v * dx
    
    # Assemble system
    a.Assemble()
    f.Assemble()
        
    # 5. Solve for the free dofs
    gfu.vec.data = a.mat.Inverse(Vh.FreeDofs()) * f.vec
    l2error = sqrt(Integrate((gfu - exactu)**2 * dx, mesh))
    return l2error, gfu.space.ndof

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
# Quadrilateral (or simplicial mesh)
quad_mesh = False
# Finite element space order
order = 1
# Stabilization parameter for ghost-penalty
gamma_stab = 0.1
# Stabilization parameter for Nitsche
lambda_nitsche = 10 * order * order

r2 = 3 / 4  # outer radius
r1 = 1 / 4  # inner radius
rc = (r1 + r2) / 2.0
rr = (r2 - r1) / 2.0
r = sqrt(x**2 + y**2)
levelset = IfPos(r - rc, r - rc - rr, rc - r - rr) 
# IfPos(cond, val1, val2) 是一个条件函数，如果 cond > 0 返回 val1，否则返回 val2。

exactu = (20 * (r2 - sqrt(x**2 + y**2)) * (sqrt(x**2 + y**2) - r1)).Compile()
coeff_f = - (exact.Diff(x).Diff(x) + exact.Diff(y).Diff(y)).Compile()

results = []

for k in range(1, 5):
    mh = 1/2**k
    l2error, ndof = def solve_poisson_unfitted(mh, quad_mesh, order, levelset, ifghost, lambda_nitsche, gamma_stab, coeff_f, exactu )
    results.append((mh,ndof,l2error))


print_convergence_table(results)
