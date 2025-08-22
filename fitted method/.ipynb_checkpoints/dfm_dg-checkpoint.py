from math import pi,exp 
from ngsolve import *
from xfem import *
from netgen.occ import *
from ngsolve.webgui import *



def solve_dfm_dg(mh,order,sigma,alpha_gamma,beta_gamma,u_exact,gD):
    # Define the bulk and the fracture region
    Omega = Rectangle(1, 1).Face() 
    Omega.faces.name = "Omega"
    Omega.edges.Min(X).name = "left"
    Omega.edges.Min(Y).name = "bottom"
    Omega.edges.Max(X).name = "right"
    Omega.edges.Max(Y).name = "top"
    fracture = Segment((0,1,0),(1,0,0))
    geo = Glue([Omega, fracture])
    
    mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=mh))
    mesh.ngmesh.SetBCName(1,"fracture")

    levelset = x+y-1
    lsetp1 = GridFunction(H1(mesh,order=1))
    InterpolateToP1(levelset,lsetp1)

    ci = CutInfo(mesh,lsetp1)
    gamma_facets = GetFacetsWithNeighborTypes(mesh, a=ci.GetElementsOfType(HASNEG), b=ci.GetElementsOfType(POS),bnd_val_a=False,bnd_val_b=False,use_and=True)
    ba_surround_facets = GetElementsWithNeighborFacets(mesh,gamma_facets)

    # Construct the unfitted fem space 
    Vh = L2(mesh,order=order,dgjumps=True)
    Vhfbase = H1(mesh,order=order,dirichlet='top|bottom|left|right')
    Vhf = Compress(Vhfbase,GetDofsOfElements(Vhfbase,ci.GetElementsOfType(IF)))
    Xh = FESpace([Vh,Vhf])

    u,v = Xh.TnT()
    jump_u = u[0] - u[0].Other() 
    jump_v = v[0] - v[0].Other() 
    n = specialcf.normal(2)
    mean_dudn = 0.5*n*(grad(u[0])+grad(u[0].Other()))
    mean_dvdn = 0.5*n*(grad(v[0])+grad(v[0].Other()))
    
    mean_u_gamma = (u[0]+u[0].Other())/2
    mean_v_gamma = (v[0]+v[0].Other())/2

    gamma_facets = GetFacetsWithNeighborTypes(mesh, a=ci.GetElementsOfType(HASNEG), b=ci.GetElementsOfType(POS),bnd_val_a=False,bnd_val_b=False,use_and=True)
    dgamma = dx(skeleton=True, definedonelements=gamma_facets)
    
    Ah = BilinearForm(Xh)
    
    # A^{DG} term
    Ah += grad(u[0])*grad(v[0])*dx
    Ah += -(mean_dvdn * jump_u + mean_dudn * jump_v)*dx(skeleton=True,definedon=mesh.Materials("Omega")) \
            + (mean_dvdn * jump_u + mean_dudn * jump_v)*dgamma
    Ah += sigma * jump_u*jump_v*dx(skeleton=True,definedon=mesh.Materials("Omega")) - sigma * jump_u*jump_v*dgamma
    Ah += -(grad(v[0]).Trace()*n*u[0] - grad(u[0]).Trace()*n*v[0])*ds(skeleton=True,definedon=mesh.Boundaries("left|right|top|bottom"))
    Ah += sigma*u[0]*v[0]*ds(skeleton=True,definedon=mesh.Boundaries("left|right|top|bottom"))
    
    # # A_gamma term
    Ah += grad(u[1]).Trace()*grad(v[1]).Trace()*ds(definedon=mesh.Boundaries("fracture"))
    
    # I^{DG} term
    Ah += beta_gamma*jump_u*jump_v *dgamma
    Ah += alpha_gamma * (mean_u_gamma-u[1])*(mean_v_gamma-v[1]) *dgamma
    Ah.Assemble()

    Fh = LinearForm(Xh)
    Fh += f[0] * v[0] * dx(definedon=mesh.Materials("Omega").Split()[0]) \
          + f[1] * v[0] * dx(definedon=mesh.Materials("Omega").Split()[1]) \
          + f[2] * v[1] * ds(definedon=mesh.Boundaries("fracture"))
    # Dirichlet boundary term
    Fh += sigma * v[0] * gD[0] * ds(skeleton=True,definedon=mesh.Boundaries('left|bottom')) \
          + sigma * v[0] * gD[1] * ds(skeleton=True,definedon=mesh.Boundaries('right|top'))
    Fh += -grad(v[0])*n*gD[0]*ds(skeleton=True,definedon=mesh.Boundaries('left|bottom')) \
          -grad(v[0])*n*gD[1]*ds(skeleton=True,definedon=mesh.Boundaries('right|top')) \
    
    Fh.Assemble()

    uh = GridFunction(Xh)
    uh.components[1].Set(u_exact[2],BND)
    fh = Fh.vec.CreateVector()
    fh.data = Fh.vec - Ah.mat * uh.vec
    uh.vec.data += Ah.mat.Inverse(Xh.FreeDofs())*fh

    err_bulk = sqrt(Integrate((uh.components[0] - u_exact[0])**2*dx(definedon=mesh.Materials("Omega").Split()[0])+(uh.components[0] - u_exact[1])**2*dx(definedon=mesh.Materials("Omega").Split()[1]), mesh=mesh))
    err_fracture = sqrt(Integrate((uh.components[1] - u_exact[2])**2*ds(definedon=mesh.Boundaries("fracture")), mesh=mesh))


    return uh.space.ndof,err_bulk,err_fracture

def print_convergence_table(results):
    print(f"{'h':>8} | {'DoFs':>8} | {'L2 Error':>12} | {'Order':>6} | {'L2 Error':>12} | {'Order':>6}")
    print("-" * 45)
    for i, (h, dofs, errb, errf) in enumerate(results):
        if i == 0:
            print(f"{h:8.4f} | {dofs:8d} | {errb:12.4e} | {'-':>6} | {errf:12.4e} | {'-':>6}")
        else:
            prev_h, _, prev_errorb, prev_errorf = results[i-1]
            rate1 = (np.log(prev_errorb) - np.log(errb)) / (np.log(prev_h) - np.log(h))
            rate2 = (np.log(prev_errorf) - np.log(errf)) / (np.log(prev_h) - np.log(h))
            print(f"{h:8.4f} | {dofs:8d} | {errb:12.4e} | {rate1:6.2f} | {errf:12.4e} | {rate2:6.2f}")


# Define important parameters



sigma = 10**3 # penalty parameter
lf = 0.01 # the aperture of the fracture
kf = 1
kfn = 1
eta_gamma = lf/kfn
ksi = 1
order = 1
# f = [-2*exp(x+y),-exp(x+y),exp(1)/sqrt(2)]
# beta_gamma = 1/eta_gamma/2
# alpha_gamma = 2/eta_gamma/(2*ksi-1)
# u_exact = [exp(x+y),exp(x+y)/2 + exp(1)*(1/2 + 3*eta_gamma/sqrt(2)),exp(1)*(1+sqrt(2)*eta_gamma)]

beta_gamma = 1/eta_gamma
alpha_gamma = 4/eta_gamma/(2*ksi-1)
u_exact = [exp(x+y),exp(x+y)/2 + exp(1)*(1/2 + 3*eta_gamma/sqrt(2)/2),exp(1)*(1+sqrt(2)/2*eta_gamma)]
gD = u_exact

results = []

for k in range(1, 5):
    mh = 1/2**k
    ndof, err_bulk, err_fracture = solve_dfm_dg(mh,order,sigma,alpha_gamma,beta_gamma,u_exact,gD)
    # L2误差计算
    results.append((mh,ndof,err_bulk, err_fracture))

print_convergence_table(results)