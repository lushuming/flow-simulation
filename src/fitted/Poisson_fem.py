from ngsolve import *
from netgen.geom2d import unit_square # mesh几何
from ngsolve.webgui import Draw

def solve_poisson(f, gD, dirichlet_bdr, gN=None, neumann_bdr=None, order=2, mesh=None, h=0.1):
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
    if mesh is None:
        mesh = Mesh(unit_square.GenerateMesh(maxh=h))

    # 2. Construct the finite element space
    fes = H1(mesh,order=order,dirichlet=dirichlet_bdr)

    # 3. Extension of Dirichlet boundary condition
    gfu = GridFunction(fes)
    gfu.set(gD,BND)

    # 4. Forms and Assembly
    u,v = fes.TNT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    rhs = LinearForm(f*v*dx).Assemble()
    r = rhs.vec - a.mat*gfu.vec
    
    # 5. Solve for the free dofs
    gfu.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs())*r

    Draw(gfu);

