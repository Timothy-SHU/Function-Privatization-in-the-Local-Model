import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import sici
from scipy.stats import laplace, gamma
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv, pinv, sqrtm, block_diag
from scipy.integrate import quad, quad_vec, IntegrationWarning
from pathos.multiprocessing import ProcessingPool

import logging
import warnings
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.INFO)
warnings.simplefilter("ignore", category = IntegrationWarning)      # ignore integration precision warnings

INTLIM = 1000
INTLIM_PER_PIECE = 100

class PrivatePiecewiseApprox:
    def __init__(self, interval, breakpoints, basis_type, degree = 1, parallel = False):
        self.degree = degree
        self.l, self.r = np.float64(interval)
        self.breakpoints = breakpoints
        self.basis_type = basis_type
        self.isOrthonormal = False

        self.basis = []
        self.params = []
        for i in range(len(breakpoints)-1):
            self.basis.append([])
            self.params.append([])
            for k in range(degree+1):
                if basis_type == 'Linear-2D':
                    self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k, 0))
                    self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k, 1))
                    self.params[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k, -1))
                else:
                    self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k))
                    if basis_type == 'Fourier' and k > 0:
                        self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], -k))
                    if basis_type == 'Sinc' or basis_type == 'Sinc-unbounded':
                        self.params[-1].append({'a': breakpoints[i]+k})
        
        self.m = len(breakpoints)-1    # m pieces in total
        self.d = len(self.basis[0])    # each piece has d basis functions
        if basis_type == 'Polynomial' and degree <= 2:
            self.isOrthonormal = True
        elif basis_type == 'Linear-2D':
            self.isOrthonormal = True
        elif basis_type == 'Sinc-unbounded':
            self.isOrthonormal = True
        else:
            self.G = np.zeros((self.m, self.d, self.d))
            self.invG = np.zeros((self.m, self.d, self.d))
            if not parallel:
                for i in range(self.m):
                    for j1 in range(self.d):
                        for j2 in range(self.d):
                            l = breakpoints[i]; r = breakpoints[i+1]
                            task = (i, j1, j2, self._get_basis_integrand(i, j1, j2), l, r)
                            _, _, _, self.G[i, j1, j2] = self._basis_integrate_task(task)
                    self.invG[i] = pinv(self.G[i])
                    # np.set_printoptions(precision = 4, linewidth = 100, suppress = True)
                    # logging.info("G = \n%s", self.G[i])
                    # logging.info("inv(G) = \n%s", self.invG[i])
                    # if np.any(np.linalg.eigvals(self.G[i]) < -1e-8):
                    #     logging.error("ERR: the pairwise inner product matrix is not SPSD!")
            else:
                tasks = []
                for i in range(self.m):
                    for j1 in range(self.d):
                        for j2 in range(self.d):
                            l = breakpoints[i]; r = breakpoints[i+1]
                            tasks.append((i, j1, j2, self._get_basis_integrand(i, j1, j2), l, r))
                with ProcessingPool(ncpus = 12) as pool:
                    results = pool.map(self._basis_integrate_task, tasks)
                for i, j1, j2, integral in results:
                    self.G[i, j1, j2] = integral
                for i in range(self.m):
                    self.invG[i] = pinv(self.G[i])
    
    def createBasis(self, l, r, k, dim = 0):
        if self.basis_type == 'Polynomial':
            if k == 0:
                return lambda x: 1/np.sqrt(r-l)
            elif k == 1:
                return lambda x: np.sqrt(12/((r-l)**3))*(x-(l+r)/2)
            elif k == 2:
                return lambda x: np.sqrt(180/((r-l)**5))*((x-(l+r)/2)**2)-np.sqrt(5/(r-l))/2
            else:
                return lambda x: np.sqrt((2*k+1)/2) * 1/np.sqrt(((r-l)/2)**(2*k+1)) * ((x-(l+r)/2)**k)
        elif self.basis_type == 'Linear-2D':
            if k == 0 and dim == 0:
                return lambda t: np.array([1/np.sqrt(r-l), 0])
            elif k == 0 and dim == 1:
                return lambda t: np.array([0, 1/np.sqrt(r-l)])
            elif k == 0 and dim == -1:  # param
                return {'k': 0, 'b': 1/np.sqrt(r-l)}
            elif k == 1 and dim == 0:
                return lambda t: np.array([np.sqrt(12/((r-l)**3))*(t-(l+r)/2), 0])
            elif k == 1 and dim == 1:
                return lambda t: np.array([0, np.sqrt(12/((r-l)**3))*(t-(l+r)/2)])
            elif k == 1 and dim == -1:
                return {'k': np.sqrt(12/((r-l)**3)), 'b': -np.sqrt(12/((r-l)**3))*(l+r)/2}
            else:
                logging.error(f"ERR: linear 2D basis param 'k = {k}, dim = {dim}' not valid.")
        elif self.basis_type == 'Fourier':
            if k == 0:
                return lambda x: 1
            elif k > 0:
                return lambda x: np.sin(2*np.pi*k*(x-(l+r)/2))
            else:
                return lambda x: np.cos(-2*np.pi*k*(x-(l+r)/2))
        elif self.basis_type == 'Sinc' or self.basis_type == 'Sinc-unbounded':
            return lambda x: np.sinc(x-l-k)
        else:
            logging.error(f"ERR: no such basis '{self.basis_type}'.")

    def _get_basis_integrand(self, i, j1, j2):
        if self.basis_type == 'Sinc': return None
        return lambda x: self.basis[i][j1](x)*self.basis[i][j2](x)
    def _basis_integrate_task(self, task):
        if self.basis_type == 'Sinc':
            l = task[4]; r = task[5]
            a1 = self.params[task[0]][task[1]]['a']
            a2 = self.params[task[0]][task[2]]['a']
            if a1 == a2:
                integral = sici(2*np.pi*(r-a1))[0] / np.pi
                if r != a1: integral += (np.cos(np.pi*(r-a1))**2-1) / ((np.pi**2)*(r-a1))
                integral -= sici(2*np.pi*(l-a1))[0] / np.pi
                if l != a1: integral -= (np.cos(np.pi*(l-a1))**2-1) / ((np.pi**2)*(l-a1))
            else:
                integral = np.sin(np.pi*(a2-a1)) * (sici(2*np.pi*(r-a2))[0] + sici(2*np.pi*(r-a1))[0])
                # integral -= np.cos(np.pi*(a2-a1)) * (np.log(np.abs(r-a1)/np.abs(r-a2))
                #                                     + sici(2*np.pi*(r-a2))[1] - sici(2*np.pi*(r-a1))[1])
                if r == a1: integral += np.cos(np.pi*(a2-a1)) * (np.euler_gamma+np.log(2*np.pi))
                else: integral += np.cos(np.pi*(a2-a1)) * (sici(2*np.pi*(r-a1))[1] - np.log(np.abs(r-a1)))
                if r == a2: integral -= np.cos(np.pi*(a2-a1)) * (np.euler_gamma+np.log(2*np.pi))
                else: integral -= np.cos(np.pi*(a2-a1)) * (sici(2*np.pi*(r-a2))[1] - np.log(np.abs(r-a2)))
                integral -= np.sin(np.pi*(a2-a1)) * (sici(2*np.pi*(l-a2))[0] + sici(2*np.pi*(l-a1))[0])
                # integral += np.cos(np.pi*(a2-a1)) * (np.log(np.abs(l-a1)/np.abs(l-a2))
                #                                     + sici(2*np.pi*(l-a2))[1] - sici(2*np.pi*(l-a1))[1])
                if l == a1: integral -= np.cos(np.pi*(a2-a1)) * (np.euler_gamma+np.log(2*np.pi))
                else: integral -= np.cos(np.pi*(a2-a1)) * (sici(2*np.pi*(l-a1))[1] - np.log(np.abs(l-a1)))
                if l == a2: integral += np.cos(np.pi*(a2-a1)) * (np.euler_gamma+np.log(2*np.pi))
                else: integral += np.cos(np.pi*(a2-a1)) * (sici(2*np.pi*(l-a2))[1] - np.log(np.abs(l-a2)))
                integral /= 2*(np.pi**2)*(a2-a1)
        else:
            integral, _ = quad(task[3], task[4], task[5], limit = INTLIM_PER_PIECE)
        return (task[0], task[1], task[2], integral)
    def _get_fit_integrand(self, i, j):
        if self.basis_type == 'Linear-2D': return None
        return lambda x: self.func(x)*self.basis[i][j](x)
    def _fit_integrate_task(self, task):
        if self.basis_type in ['Linear-2D', 'Sinc', 'Sinc-unbounded']:
            basis_l = task[3]; basis_r = task[4]
            lidx = max(np.searchsorted(self.ts_t, basis_l, side = 'right')-1, 0)
            ridx = min(np.searchsorted(self.ts_t, basis_r, side = 'left'), len(self.ts_t)-1)
            integral = 0
            for i in range(lidx, ridx):
                l = max(basis_l, self.ts_t[i])
                r = min(basis_r, self.ts_t[i+1])
                if l >= r: continue
                if self.basis_type == 'Linear-2D':
                    k1 = (self.ts_val[i+1][task[1]%2]-self.ts_val[i][task[1]%2])/(r-l)
                    b1 = self.ts_val[i][task[1]%2]-k1*l
                    k2 = self.params[task[0]][task[1]//2]['k']
                    b2 = self.params[task[0]][task[1]//2]['b']
                    # integrate (k1*t+b1)*(k2*t+b2) on [l, r]
                    # \int k1*k2*t^2+(k1*b2+b1*k2)*t+b1*b2
                    # = 1/3*k1*k2*x^3+1/2*(k1*b2+b1*k2)*t^2+b1*b2*t
                    integral += 1/3*k1*k2*(r**3)+1/2*(k1*b2+k2*b1)*(r**2)+(b1*b2)*r
                    integral -= 1/3*k1*k2*(l**3)+1/2*(k1*b2+k2*b1)*(l**2)+(b1*b2)*l
                else:
                    k = (self.ts_val[i+1]-self.ts_val[i])/(r-l)
                    b = self.ts_val[i]-k*l
                    a = self.params[task[0]][task[1]]['a']
                    # integrate sinc(x-a)*(kx+b) on [l, r]
                    # \int sin(pi(x-a))/(pi(x-a))*(kx+b)
                    # = [pi*(ak+b)*Si(pi(x-a)) - k*cos(pi(x-a))] / (pi^2)
                    integral += (np.pi*(a*k+b)*sici(np.pi*(r-a))[0] - k*np.cos(np.pi*(r-a))) / (np.pi**2)
                    integral -= (np.pi*(a*k+b)*sici(np.pi*(l-a))[0] - k*np.cos(np.pi*(l-a))) / (np.pi**2)
        else:
            integral, _ = quad(task[2], task[3], task[4], limit = INTLIM_PER_PIECE)
        return (task[0], task[1], integral)

    def fit(self, func, time_series = None, parallel = False):
        self.func = func
        if time_series != None:
            self.ts_t, self.ts_val = time_series
        self.eps = 0
        self.method = None
        self.noise = np.zeros((self.m, self.d))
        
        if time_series != None:
            self.funcSqrInt = 0
            for i in range(len(self.ts_t)-1):
                l = self.ts_t[i]; r = self.ts_t[i+1]
                if l >= r: continue
                if self.basis_type == 'Linear-2D':
                    for j in range(0, 2):
                        k = (self.ts_val[i+1][j]-self.ts_val[i][j])/(r-l)
                        b = self.ts_val[i][j]-k*l
                        # integrate (kt+b)^2 on [l, r]
                        # \int k^2t^2+2kbt+b^2 = 1/3*k^2t^3+kbt^2+b^2*t
                        self.funcSqrInt += 1/3*(k**2)*(r**3)+k*b*(r**2)+(b**2)*r
                        self.funcSqrInt -= 1/3*(k**2)*(l**3)+k*b*(l**2)+(b**2)*l
                else:
                    k = (self.ts_val[i+1]-self.ts_val[i])/(r-l)
                    b = self.ts_val[i]-k*l
                    self.funcSqrInt += 1/3*(k**2)*(r**3)+k*b*(r**2)+(b**2)*r
                    self.funcSqrInt -= 1/3*(k**2)*(l**3)+k*b*(l**2)+(b**2)*l
        else:
            integrand = lambda x: func(x)**2
            self.funcSqrInt, _ = quad(integrand, self.l, self.r, limit = INTLIM)

        self.b = np.zeros((self.m, self.d))
        if not parallel:
            for i in range(self.m):
                for j in range(self.d):
                    l = self.breakpoints[i]; r = self.breakpoints[i+1]
                    task = (i, j, self._get_fit_integrand(i, j), l, r)
                    _, _, self.b[i, j] = self._fit_integrate_task(task)
        else:
            tasks = []
            for i in range(self.m):
                for j in range(self.d):
                    l = self.breakpoints[i]; r = self.breakpoints[i+1]
                    tasks.append((i, j, self._get_fit_integrand(i, j), l, r))
            with ProcessingPool() as pool:
                results = pool.map(self._fit_integrate_task, tasks)
            for i, j, integral in results:
                self.b[i, j] = integral

        if self.isOrthonormal:
            self.coeff = self.b
        else:
            self.coeff = np.zeros((self.m, self.d))
            for i in range(self.m):
                self.coeff[i] = self.invG[i]@self.b[i]
        # logging.info("b = %s", self.b)
        # logging.info("c = %s", self.coeff)

    def privatize(self, eps = 0.5, method = 'Laplace'):
        self.eps = eps
        self.method = method
        if method == None:
            self.noise = np.zeros((self.m, self.d))
        elif method == 'Laplace':
            self.noise = np.random.randn(self.m*self.d)
            self.noise = self.noise/np.linalg.norm(self.noise)
            self.noise = self.noise.reshape(self.m, self.d)
            self.noise *= gamma.rvs(a = self.m*self.d, scale = 1/eps)
            if not self.isOrthonormal:
                for i in range(self.m):
                    self.noise[i] = (sqrtm(self.invG[i])@self.noise[i]).real
        elif method == 'Normal':
            self.noise = np.random.randn(self.m*self.d)
            self.noise = self.noise.reshape(self.m, self.d)
            if not self.isOrthonormal:
                self.noise[i] = (sqrtm(self.invG[i])@self.noise[i]).real
            self.noise /= np.sqrt(2*eps)
        else:
            logging.error(f"ERR: no such method '{method}'.")
        # logging.info("noise = %s", self.noise)

    def smooth(self, method = 'Sparse-KKT'):
        if self.m == 1: return
        if self.isOrthonormal:
            M = np.eye(self.m*self.d)
        else:
            M = block_diag(*self.G)
        q = -(self.coeff+self.noise).flatten()@M
        if self.basis_type == 'Linear-2D':
            C = np.zeros(((self.m-1)*2, self.m*self.d))
        else:
            C = np.zeros((self.m-1, self.m*self.d))
        for i in range(self.m-1):
            for j in range(self.d):
                if self.basis_type == 'Linear-2D':
                    C[i*2, i*self.d+j], C[i*2+1, i*self.d+j] = self.basis[i][j](self.breakpoints[i+1])
                    C[i*2, (i+1)*self.d+j], C[i*2+1, (i+1)*self.d+j] = -self.basis[i+1][j](self.breakpoints[i+1])
                else:
                    C[i, i*self.d+j] = self.basis[i][j](self.breakpoints[i+1])
                    C[i, (i+1)*self.d+j] = -self.basis[i+1][j](self.breakpoints[i+1])
        if method == 'CVXPY':
            y = cp.Variable(self.m*self.d)
            QP = cp.Problem(cp.Minimize((1/2)*cp.quad_form(y, M)+q@y), 
                            [C@y == np.zeros(C.shape[0])])
            QP.solve()
            for i in range(self.m):
                for j in range(self.d):
                    self.noise[i, j] = y.value[i*self.d+j]-self.coeff[i, j]
        elif method == 'Sparse-KKT':
            n = M.shape[0]; m = C.shape[0]
            M_sp = sparse.csc_matrix(M)
            C_sp = sparse.csc_matrix(C)
            KKT_upper = sparse.hstack([M_sp, C_sp.T])
            KKT_lower = sparse.hstack([C_sp, sparse.csc_matrix((m, m))])
            KKT = sparse.vstack([KKT_upper, KKT_lower]).tocsc()
            RHS = np.concatenate([-q, np.zeros(C.shape[0])])
            y = spsolve(KKT, RHS)[:n]
            for i in range(self.m):
                for j in range(self.d):
                    self.noise[i, j] = y[i*self.d+j]-self.coeff[i, j]
        else:
            print(f"ERR: no such method '{method}'.")
    
    def createApprox(self):
        if self.func == None:
            logging.error(f"ERR: solver not fitted but called for approximation result.")
        def approx(x):
            if isinstance(x, (int, float)):
                ret = 0
                if x >= self.l and x <= self.r:
                    i = min(np.searchsorted(self.breakpoints, x, side = 'right')-1, self.m-1)
                    for j in range(self.d):
                        ret += self.coeff[i, j]*self.basis[i][j](x)
                return ret
            else:
                return np.array([approx(e) for e in x])
        return approx

    def createPriv(self):
        if self.func == None:
            logging.error(f"ERR: solver not fitted but called for privatization result.")
        def priv(x):
            if isinstance(x, (int, float)):
                ret = 0
                if x >= self.l and x <= self.r:
                    i = min(np.searchsorted(self.breakpoints, x, side = 'right')-1, self.m-1)
                    for j in range(self.d):
                        ret += (self.coeff[i, j]+self.noise[i, j])*self.basis[i][j](x)
                return ret
            else:
                return np.array([priv(e) for e in x])
        return priv

    def eval(self, type = 'Approx'):
        # Evaluate l2-dist between func and approx / priv
        # \int(func - \sum_i(coeff[i]*basis[i]))^2
        # = func^2 - 2*\sum_i(coeff[i]*func*basis[i]) 
        #   + \sum_i\sum_j(coeff[i]*coeff[j]*basis[i]*basis[j])
        # The second term is 2*coeff@b
        # The third term is coeff^T @ G @ coeff
        err = self.funcSqrInt
        if type == 'Approx':
            for i in range(self.m):
                err -= 2*self.coeff[i]@self.b[i]
                if self.isOrthonormal:
                    err += self.coeff[i]@self.coeff[i]
                else:
                    err += self.coeff[i].T@self.G[i]@self.coeff[i]
            if err < 0:
                approx = self.createApprox()
                integrand = lambda x: (self.func(x)-approx(x))**2
                if self.basis_type == 'Linear-2D':
                    integral_vec, _ = quad_vec(integrand, self.l, self.r, limit = INTLIM)
                    err = np.sum(integral_vec)
                else:
                    err, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        elif type == 'Priv':
            for i in range(self.m):
                err -= 2*(self.coeff[i]+self.noise[i])@self.b[i]
                if self.isOrthonormal:
                    err += (self.coeff[i]+self.noise[i])@(self.coeff[i]+self.noise[i])
                else:
                    err += (self.coeff[i]+self.noise[i]).T@self.G[i]@(self.coeff[i]+self.noise[i])
            if err < 0:
                priv = self.createPriv()
                integrand = lambda x: (self.func(x)-priv(x))**2
                if self.basis_type == 'Linear-2D':
                    integral_vec, _ = quad_vec(integrand, self.l, self.r, limit = INTLIM)
                    err = np.sum(integral_vec)
                else:
                    err, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        else:
            logging.error(f"ERR: no such loss type '{type}'.")
        return np.sqrt(err)

    def evalPrivLoss(self):
        # Evaluate l2-dist between approx and priv
        # \int(\sum_i(noise[i]*basis[i]))^2)
        # which is just noise^T @ G @ noise
        err = 0
        if self.isOrthonormal:
            for i in range(self.m):
                err += self.noise[i]@self.noise[i]
        else:
            for i in range(self.m):
                err += self.noise[i].T@self.G[i]@self.noise[i]
        return np.sqrt(err)

def time_series_func(t, val):
    def func(x):
        return np.interp(x, t, val)
    return func

def time_series_func_2D(t, val1, val2):
    def func2D(x):
        return np.array([np.interp(x, t, val1), np.interp(x, t, val2)])
    return func2D

def plot_1D_1D(solver):
    x_plot = np.linspace(solver.l, solver.r, INTLIM)
    y_true = solver.func(x_plot)
    y_approx = solver.createApprox()(x_plot)
    y_priv = solver.createPriv()(x_plot)
    err_ls = solver.eval('Approx')
    err_priv = solver.eval('Priv')
    plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'Func')
    plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Approx')
    plt.plot(x_plot, y_priv, 'b--', linewidth = 1.5, label = 'Priv Approx')
    plt.title(f"{solver.basis_type} basis, degree = {solver.degree}, error: {err_priv:.5f} ({err_ls:.5f})")
    plt.legend(loc = 'upper left')
    plt.grid(True)
