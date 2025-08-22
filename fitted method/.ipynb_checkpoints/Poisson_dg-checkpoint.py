from ngsolve import *
from netgen.geom2d import unit_square # mesh几何
from ngsolve.webgui import Draw
import numpy as np

def solve_poisson_dg(f, gD, dirichlet_bnd, gN=None, neumann_bnd=None, epi=-1, sigma=10, order=1, alpha=1, mh=0.1):
    """
    解决带有非齐次Dirichlet和Neumann边界条件的Poisson问题:
        -Δu = f  在域内
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

    # 2. Define the DG space 
    fes = L2(mesh,order=order,dgjumps=True)
    u,v = fes.TnT()

    # 3. Define the jumps and the averages
    jump_u = u - u.Other()
    jump_v = v - v.Other()
    n = specialcf.normal(2)
    mean_dudn = 0.5*n*(grad(u)+grad(u.Other()))
    mean_dvdn = 0.5*n*(grad(v)+grad(v.Other()))
    h = specialcf.mesh_size   

    # 4. Define bilinear form
    diffusion = grad(u)*grad(v)*dx + alpha*u*v*dx \
                + (epi * mean_dvdn * jump_u- mean_dudn * jump_v)*dx(skeleton=True) \
                + sigma/h* jump_u*jump_v*dx(skeleton=True) \
                + (epi * grad(v).Trace()*n*u - grad(u).Trace()*n*v)*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))  \
                + sigma/h*u*v*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd))
    a_epi = BilinearForm(diffusion).Assemble()
    # Define the right hand side
    rl = f * v*dx \
        + (epi*grad(v).Trace()*n + sigma/h*v)*gD*ds(skeleton=True,definedon=mesh.Boundaries(dirichlet_bnd)) \
        + v*gN*ds(skeleton=True,definedon=mesh.Boundaries(neumann_bnd))
    F = LinearForm(rl).Assemble()
        
    # 5. Solve for the free dofs
    gfu = GridFunction(fes,name="uDG")
    gfu.vec.data = a_epi.mat.Inverse()*F.vec
    return gfu,mesh


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
err = np.zeros(4,1)
for k in range(1, 5):
    mh = 1/2^k
    gfu,mesh = solve_poisson_dg(f, gD, dirichlet_bnd, gN, neumann_bnd, epi, sigma, order, alpha, mh)
    # L2误差计算
    u_exact = sin(x)*sin(y)
    err[k-1] = sqrt(Integrate((gfu - u_exact)**2, mesh))
print(err)


