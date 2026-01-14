from dataclasses import dataclass
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve.internal import *
from xfem import *
from xfem.lsetcurv import *
from math import pi
import numpy as np

# -------------------------
# Parameter dataclasses
# -------------------------
@dataclass
class PhysicalParams:
    mu: float = 10.0
    lam: float = 100.0
    alpha: float = 1.0
    Kp: float = 1.0
    Kft: float = 100.0
    Kfn: float = 100.0
    d: float = 1e-4
    ksi: float = 0.75

@dataclass
class NumericalParams:
    order_eta: int = 2
    order_p: int = 1
    beta_eta: float = 300.0
    beta_p: float = 300.0
    sigma_eta: float = 0.1
    sigma_p: float = 0.1
    mh: float = 0.25

# -------------------------
# Solver class
# -------------------------
class BiotDarcySolver:
    def __init__(self, phys: PhysicalParams, num: NumericalParams):
        self.phys = phys
        self.num = num
        self.mesh = None
        self.lsetp1 = None
        self.ci = None
        self.fes = None
        self.ah = None
        self.fh = None
        self.gfu = None

    # Stress as a method to avoid global mu/lam
    def Stress(self, strain):
        mu = self.phys.mu
        lam = self.phys.lam
        return 2*mu*strain + lam*Trace(strain)*Id(2)

    def build_mesh(self):
        Omega = Rectangle(1, 1).Face()
        Omega.faces.name = "Omega"
        Omega.edges.name = "outer"
        self.mesh = Mesh(OCCGeometry(Omega, dim=2).GenerateMesh(maxh=self.num.mh))

    def build_levelset_and_cutinfo(self, levelset):
        # levelset is a CoefficientFunction or expression
        self.lsetp1 = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(levelset, self.lsetp1)
        self.ci = CutInfo(self.mesh, self.lsetp1)

    def build_spaces(self):
        ci = self.ci
        mesh = self.mesh
        order_eta = self.num.order_eta
        order_p = self.num.order_p

        Ehbase = VectorL2(mesh, order=order_eta, dirichlet="outer", dgjumps=True)
        Phbase = L2(mesh, order=order_p, dirichlet="outer", dgjumps=True)
        Fhbase = H1(mesh, order=order_p, dirichlet=".*", dgjumps=True)

        haspos = ci.GetElementsOfType(HASPOS)
        hasneg = ci.GetElementsOfType(HASNEG)
        hasif = ci.GetElementsOfType(IF)

        E1 = Compress(Ehbase, GetDofsOfElements(Ehbase, hasneg))
        E2 = Compress(Ehbase, GetDofsOfElements(Ehbase, haspos))
        P1 = Compress(Phbase, GetDofsOfElements(Phbase, hasneg))
        P2 = Compress(Phbase, GetDofsOfElements(Phbase, haspos))
        Pf = Compress(Fhbase, GetDofsOfElements(Fhbase, hasif))

        self.fes = E1*E2*P1*P2*Pf

    def assemble(self, levelset, fe, fp, ff, exact_eta, exact_p, exact_f):
        # build mesh, cutinfo, spaces if not already
        if self.mesh is None:
            self.build_mesh()
        self.build_levelset_and_cutinfo(levelset)
        self.build_spaces()

        fes = self.fes
        (eta1,eta2,p1,p2,pf), (xi1,xi2,q1,q2,qf) = fes.TnT()

        # local aliases for parameters
        phys = self.phys
        num = self.num
        mu = phys.mu; lam = phys.lam; alpha = phys.alpha
        Kp = phys.Kp; Kft = phys.Kft
        beta_eta = num.beta_eta; beta_p = num.beta_p
        sigma_eta = num.sigma_eta; sigma_p = num.sigma_p

        # special variables
        h = specialcf.mesh_size
        nf = Normalize(grad(self.lsetp1))
        ne = specialcf.normal(2)

        # jumps and averages
        jump_eta1 = eta1 - eta1.Other()
        jump_eta2 = eta2 - eta2.Other()
        jump_xi1 = xi1 - xi1.Other()
        jump_xi2 = xi2 - xi2.Other()
        jump_p1 = p1 - p1.Other()
        jump_p2 = p2 - p2.Other()
        jump_q1 = q1 - q1.Other()
        jump_q2 = q2 - q2.Other()
        jump_pf = pf - pf.Other()
        jump_qf = qf - qf.Other()

        strain_eta1 = Sym(Grad(eta1))
        strain_eta2 = Sym(Grad(eta2))
        strain_xi1 = Sym(Grad(xi1))
        strain_xi2 = Sym(Grad(xi2))

        # mean stresses using local Stress method
        mean_stress_eta1 = 0.5*(self.Stress(Sym(Grad(eta1))) + self.Stress(Sym(Grad(eta1.Other()))))*ne
        mean_stress_eta2 = 0.5*(self.Stress(Sym(Grad(eta2))) + self.Stress(Sym(Grad(eta2.Other()))))*ne
        mean_stress_xi1 = 0.5*(self.Stress(Sym(Grad(xi1))) + self.Stress(Sym(Grad(xi1.Other()))))*ne
        mean_stress_xi2 = 0.5*(self.Stress(Sym(Grad(xi2))) + self.Stress(Sym(Grad(xi2.Other()))))*ne

        # mean normal derivatives for pressure
        mean_dp1dn = 0.5*Kp*(grad(p1)+grad(p1.Other()))*ne
        mean_dq1dn = 0.5*Kp*(grad(q1)+grad(q1.Other()))*ne
        mean_dp2dn = 0.5*Kp*(grad(p2)+grad(p2.Other()))*ne
        mean_dq2dn = 0.5*Kp*(grad(q2)+grad(q2.Other()))*ne
        mean_dpfdn = 0.5*Kft*(grad(pf)+grad(pf.Other()))*ne
        mean_dqfdn = 0.5*Kft*(grad(qf)+grad(qf.Other()))*ne

        mean_p1 = 0.5*(p1 + p1.Other())
        mean_q1 = 0.5*(q1 + q1.Other())
        mean_p2 = 0.5*(p2 + p2.Other())
        mean_q2 = 0.5*(q2 + q2.Other())

        # integration operators
        ci = self.ci
        haspos = ci.GetElementsOfType(HASPOS)
        hasneg = ci.GetElementsOfType(HASNEG)
        hasif = ci.GetElementsOfType(IF)

        interior_neg_facets = GetFacetsWithNeighborTypes(self.mesh, a=hasneg, b=hasneg)
        interior_pos_facets = GetFacetsWithNeighborTypes(self.mesh, a=haspos, b=haspos)
        gp_neg_faces = GetFacetsWithNeighborTypes(self.mesh, a=hasneg, b=hasif)
        gp_pos_faces = GetFacetsWithNeighborTypes(self.mesh, a=haspos, b=hasif)

        dx_neg = dCut(self.lsetp1, NEG, definedonelements=hasneg)
        dx_pos = dCut(self.lsetp1, POS, definedonelements=haspos)
        dgamma = dCut(self.lsetp1, IF, definedonelements=hasif)

        dk_neg = dCut(self.lsetp1, NEG, skeleton=True, definedonelements=interior_neg_facets)
        dk_pos = dCut(self.lsetp1, POS, skeleton=True, definedonelements=interior_pos_facets)

        # domain boundary integrals: keep simple ds(skeleton=True) as in your refactor
        dso_neg = ds(skeleton=True)
        dso_pos = ds(skeleton=True)

        dw_neg = dFacetPatch(definedonelements=gp_neg_faces)
        dw_pos = dFacetPatch(definedonelements=gp_pos_faces)

        # assemble bilinear form
        ah = BilinearForm(fes)
        # Equation 1
        ah += 2*mu*InnerProduct(strain_eta1,strain_xi1)*dx_neg + lam*div(eta1)*div(xi1)*dx_neg \
                - (InnerProduct(mean_stress_eta1,jump_xi1) + InnerProduct(mean_stress_xi1,jump_eta1) - beta_eta/h*InnerProduct(jump_eta1,jump_xi1))*dk_neg \
                - (InnerProduct(self.Stress(Sym(Grad(eta1)))*ne,xi1) + InnerProduct(self.Stress(Sym(Grad(xi1)))*ne,eta1) - beta_eta/h*InnerProduct(eta1,xi1))*dso_neg
        ah += 2*mu*InnerProduct(strain_eta2,strain_xi2)*dx_pos + lam*div(eta2)*div(xi2)*dx_pos \
                - (InnerProduct(mean_stress_eta2,jump_xi2) + InnerProduct(mean_stress_xi2,jump_eta2) - beta_eta/h*InnerProduct(jump_eta2,jump_xi2))*dk_pos \
                - (InnerProduct(self.Stress(Sym(Grad(eta2)))*ne,xi2) + InnerProduct(self.Stress(Sym(Grad(xi2)))*ne,eta2) - beta_eta/h*InnerProduct(eta2,xi2))*dso_pos

        # Bm
        ah += -alpha*(div(xi1)*p1*dx_neg - mean_p1*jump_xi1*ne*dk_neg - p1*xi1*ne*dso_neg)
        ah += -alpha*(div(xi2)*p2*dx_pos - mean_p2*jump_xi2*ne*dk_pos - p2*xi2*ne*dso_pos)

        # I coupling for displacement-pressure
        ah += -alpha*pf*nf*(xi1-xi2)*dgamma

        # ghost penalty for eta
        ah += sigma_eta / (h**2) * (eta1 - eta1.Other()) * (xi1 - xi1.Other()) * dw_neg
        ah += sigma_eta / (h**2) * (eta2 - eta2.Other()) * (xi2 - xi2.Other()) * dw_pos

        # Equation 2 Ap
        ah += Kp*grad(p1)*grad(q1)*dx_neg \
                - (mean_dp1dn*jump_q1 + mean_dq1dn*jump_p1 - beta_p/h*jump_p1*jump_q1)*dk_neg \
                - (Kp*grad(p1)*ne*q1 + Kp*grad(q1)*ne*p1 - beta_p/h*p1*q1)*dso_neg
        ah += Kp*grad(p2)*grad(q2)*dx_pos \
                - (mean_dp2dn*jump_q2 + mean_dq2dn*jump_p2 - beta_p/h*jump_p2*jump_q2)*dk_pos \
                - (Kp*grad(p2)*ne*q2 + Kp*grad(q2)*ne*p2 - beta_p/h*p2*q2)*dso_pos

        # coupling term on body equations
        alphaf = phys.ksi and 4*phys.Kfn/phys.d/(2*phys.ksi-1) or 0.0
        betaf = phys.Kfn/phys.d
        ah += (alphaf *(0.5*(p1+p2) - pf)*0.5*(q1+q2) + betaf*(p1-p2)*(q1-q2))*dgamma

        # ghost penalty for pressure
        ah += sigma_p / (h**2) * (p1 - p1.Other()) * (q1 - q1.Other()) * dw_neg
        ah += sigma_p / (h**2) * (p2 - p2.Other()) * (q2 - q2.Other()) * dw_pos

        # Equation 3 Af
        ah += Kft*grad(pf)*grad(qf)*dgamma
        # coupling on fracture equation (note qf is used here)
        ah += (-alphaf *(0.5*(p1+p2) - pf)*qf + betaf*(p1-p2)*(q1-q2))*dgamma

        ah.Assemble()
        self.ah = ah

        # rhs
        fh = LinearForm(fes)
        fh += fe*xi1*dx_neg - InnerProduct(exact_eta, self.Stress(Sym(Grad(xi1)))*ne)*dso_neg + beta_eta/h*exact_eta*xi1*dso_neg
        fh += fe*xi2*dx_pos - InnerProduct(exact_eta, self.Stress(Sym(Grad(xi2)))*ne)*dso_pos + beta_eta/h*exact_eta*xi2*dso_pos

        fh += fp*q1*dx_neg - Kp*grad(q1)*ne*exact_p*dso_neg + beta_p/h*exact_p*q1*dso_neg
        fh += fp*q2*dx_pos - Kp*grad(q2)*ne*exact_p*dso_pos + beta_p/h*exact_p*q2*dso_pos

        fh += ff*qf*dgamma

        fh.Assemble()
        self.fh = fh

    def solve(self, pfD=None):
        # Solve linear system with Dirichlet on fracture if provided
        fes = self.fes
        gfu = GridFunction(fes)
        freedofs = fes.FreeDofs()
        if pfD is not None:
            # set fracture Dirichlet if needed
            gfu.components[4].Set(pfD, BND)
        self.fh.vec.data -= self.ah.mat * gfu.vec
        gfu.vec.data += self.ah.mat.Inverse(freedofs) * self.fh.vec
        self.gfu = gfu
        return gfu

    def compute_errors(self, exact_eta, exact_p, exact_f):
        # compute L2 errors similar to original
        ci = self.ci
        haspos = ci.GetElementsOfType(HASPOS)
        hasneg = ci.GetElementsOfType(HASNEG)
        dx_neg = dCut(self.lsetp1, NEG, definedonelements=hasneg)
        dx_pos = dCut(self.lsetp1, POS, definedonelements=haspos)
        dgamma = dCut(self.lsetp1, IF, definedonelements=ci.GetElementsOfType(IF))

        error_eta = sqrt(Integrate((self.gfu.components[0] - exact_eta)**2 * dx_neg + (self.gfu.components[1] - exact_eta)**2 * dx_pos, self.mesh))
        error_p = sqrt(Integrate((self.gfu.components[2] - exact_p)**2 * dx_neg + (self.gfu.components[3] - exact_p)**2 * dx_pos, self.mesh))
        error_f = sqrt(Integrate((self.gfu.components[4]-exact_f)**2*dgamma, self.mesh))
        return error_eta, error_p, error_f, self.gfu.space.ndof

# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    # define parameters
    phys = PhysicalParams(mu=10, lam=100, alpha=1, Kp=1, Kft=100, Kfn=100, d=1e-4, ksi=0.75)
    num = NumericalParams(order_eta=2, order_p=1, beta_eta=300, beta_p=300, sigma_eta=0.1, sigma_p=0.1, mh=0.25)

    # define analytic solutions and RHS as in your original script
    x, y = symbols("x y")
    eta_x = sin(pi*y) + y
    eta_y = (y-1/2)**2 - x
    exact_eta = CF((eta_x, eta_y))
    exact_p = IfPos(y-1/2, y*sin(pi*x), (y-1/(phys.Kfn/phys.d))*sin(pi*x))  # adapt if needed
    exact_f = (1-1/(phys.Kfn/phys.d))/2*sin(pi*x)

    # forcing terms (same as original)
    epsilon_xx = eta_x.Diff(x)
    epsilon_yy = eta_y.Diff(y)
    epsilon_xy = 0.5*(eta_x.Diff(y) + eta_y.Diff(x))
    sigma_xx = phys.lam*(epsilon_xx + epsilon_yy) + 2*phys.mu*epsilon_xx - phys.alpha*exact_p
    sigma_yy = phys.lam*(epsilon_xx + epsilon_yy) + 2*phys.mu*epsilon_yy - phys.alpha*exact_p
    sigma_xy = 2*phys.mu*epsilon_xy
    f_x = - (sigma_xx.Diff(x) + sigma_xy.Diff(y))
    f_y = - (sigma_xy.Diff(x) + sigma_yy.Diff(y))
    fe = CF((f_x, f_y))
    fp = -phys.Kp*(exact_p.Diff(x).Diff(x) + exact_p.Diff(y).Diff(y))
    ff = -phys.Kft*(exact_f.Diff(x).Diff(x) + exact_f.Diff(y).Diff(y))

    # levelset
    levelset = y - 1/2

    # run solver
    solver = BiotDarcySolver(phys, num)
    solver.assemble(levelset, fe, fp, ff, exact_eta, exact_p, exact_f)
    gfu = solver.solve(pfD=exact_f)
    err_eta, err_p, err_f, ndof = solver.compute_errors(exact_eta, exact_p, exact_f)
    print("Errors:", err_eta, err_p, err_f, "DoFs:", ndof)
