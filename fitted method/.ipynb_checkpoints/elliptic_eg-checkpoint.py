from ngsolve import *
from netgen.geom2d import unit_square # mesh几何
from ngsolve.webgui import Draw
import numpy as np

def solve_poisson_eg(f, gD, dirichlet_bnd, gN=None, neumann_bnd=None, epi=-1, sigma=10, order=1, alpha=1, mh=0.1):
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
    mesh = Mesh(unit_square.GenerateMesh(maxh=mh))

    # 2. Define the EG space 
    Vcg = H1(mesh, order=order)  # 连续线性空间
    Vdg = L2(mesh, order=order-1,dgjumps=True)  # 不连续常数空间（块状加密）
    fes = Vcg*Vdg

    ucg,udg = fes.TrialFunction()
    vcg,vdg = fes.TestFunction()
    a_epi = BilinearForm(fes)
    F = LinearForm(fes)
    h = specialcf.mesh_size 

    # 3. Construct the bilinear forms and the right hand side
    # 3.1 dg basis function part
    jump_u = udg - udg.Other()
    jump_v = vdg - vdg.Other()
    n = specialcf.normal(2)
    mean_dudn = 0.5*n*(grad(udg)+grad(udg.Other()))
    mean_dvdn = 0.5*n*(grad(vdg)+grad(vdg.Other()))
    a_epi += grad(udg)*grad(vdg)*dx + alpha*udg*vdg*dx \
                    + (epi * mean_dvdn * jump_u- mean_dudn * jump_v)*dx(skeleton=True) \
                    + sigma/h* jump_u*jump_v*dx(skeleton=True) \
                    + (epi * grad(vdg).Trace()*n*udg - grad(udg).Trace()*n*vdg)*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))  \
                    + sigma/h*udg*vdg*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))
    F += f * vdg*dx \
        + (epi*grad(vdg).Trace()*n + sigma/h*vdg)*gD*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd)) \
        + vdg*gN*ds(skeleton=True,definedon=mesh.Boundaries(neumann_bnd))
    
    # 3.2 cg basis function part
    a_epi += grad(ucg)*grad(vcg)*dx + alpha*ucg*vcg*dx \
                + (epi * grad(vcg).Trace()*n*ucg - grad(ucg).Trace()*n*vcg)*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))  \
                + sigma/h*ucg*vcg*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))
    F += f * vcg*dx \
        + (epi*grad(vcg).Trace()*n + sigma/h*vcg)*gD*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd)) \
        + vcg*gN*ds(skeleton=True,definedon=mesh.Boundaries(neumann_bnd))

    # 3.3 cross terms
    a_epi +=  grad(udg)*grad(vcg)*dx + alpha*udg*vcg*dx \
            + (epi * (grad(vcg)*n) * jump_u)*dx(skeleton=True) \
            + (epi * grad(vcg).Trace()*n*udg - grad(udg).Trace()*n*vcg)*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))  \
            + sigma/h*udg*vcg*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))
    a_epi += grad(ucg)*grad(vdg)*dx + alpha*ucg*vdg*dx \
                    - (grad(ucg)*n * jump_v)*dx(skeleton=True) \
                    + (epi * grad(vdg).Trace()*n*ucg - grad(ucg).Trace()*n*vdg)*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))  \
                    + sigma/h*ucg*vdg*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))

    a_epi.Assemble()
    F.Assemble()

    # 5. Solve for the free dofs
    # Solve
    gfu = GridFunction(fes,name="uEG")
    gfu.vec.data = a_epi.mat.Inverse()*F.vec
    #print(f"gfu.vec.size = {gfu.vec.size}")
    return gfu,mesh

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
epi = -1
sigma = 10
order = 1
alpha = 1
f = (2+alpha) * sin(x) * sin(y)
dirichlet_bnd = "left|right|top"
gD = sin(x) * sin(y)
neumann_bnd = 'bottom'
gN = CoefficientFunction(-sin(x) * cos(y))  # 下边界取值
results = []

for k in range(1, 5):
    mh = 1/2**k
    gfu,mesh = solve_poisson_eg(f, gD, dirichlet_bnd, gN, neumann_bnd, epi, sigma, order, alpha, mh)
    # L2误差计算
    u_exact = sin(x)*sin(y)
    uh_enriched = gfu.components[0] + gfu.components[1]
    err = sqrt(Integrate((uh_enriched - u_exact)**2, mesh))
    results.append((mh,gfu.space.ndof,err))


print_convergence_table(results)
