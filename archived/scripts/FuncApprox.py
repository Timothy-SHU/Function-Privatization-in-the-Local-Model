import numpy as np
import matplotlib.pyplot as plt
from math import log
from scipy.stats import laplace
from scipy.linalg import inv, pinv, sqrtm, solve
from scipy.integrate import quad, IntegrationWarning
import warnings

INTLIM = 1000
INTLIM_PER_PIECE = 100

class PrivateLeastSquaresApprox:
    def __init__(self, interval, breakpoints, degree = 1, weight = None):
        self.degree = degree
        self.l, self.r = interval
        self.breakpoints = breakpoints
        self.weight = weight if weight else lambda x: 1.0

        self.params = []
        for i in range(len(breakpoints)-1):
            for k in range(degree+1):
                self.params.append({'l': breakpoints[i], 'r': breakpoints[i+1], 'k': k})
        # self.basis = lambda i, x: x**self.params[i]['k'] if x >= self.params[i]['l'] and x < self.params[i]['r'] else 0
        # self.basis = lambda i, x: np.where ((x >= self.params[i]['l']) & (x <= self.params[i]['r']), 
        #     ((x-(self.params[i]['l']+self.params[i]['r'])/2) / (self.params[i]['r']-self.params[i]['l']))**self.params[i]['k'], 0)
        self.basis = lambda i, x: np.where ((x >= self.params[i]['l']) & (x <= self.params[i]['r']), 
            np.sqrt((2*self.params[i]['k']+1)/2) * 
            1/np.sqrt(((self.params[i]['r']-self.params[i]['l'])/2)**(2*self.params[i]['k']+1)) * 
            ((x-(self.params[i]['l']+self.params[i]['r'])/2)**self.params[i]['k']), 
            0)

        self.d = len(self.params)
        self.G = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                l_max = max(self.params[i]['l'], self.params[j]['l'])
                r_min = min(self.params[i]['r'], self.params[j]['r'])
                if l_max <= r_min:
                    integrand = lambda x: self.basis(i, x)*self.basis(j, x)*self.weight(x)
                    self.G[i,j], _ = quad(integrand, l_max, r_min, limit = INTLIM_PER_PIECE)
                else:
                    self.G[i,j] = 0
        # np.set_printoptions(precision = 4, linewidth = 100, suppress = True)
        # print("G =", self.G)
        # print("inv(G) =", pinv(self.G))
        if np.any(np.linalg.eigvals(self.G) < -1e-8):
            print("ERR: the pairwise inner product matrix is not SPSD!")
        
    def solve(self, func, eps = 0.5, method = 'Laplace'):
        b = np.zeros(self.d)
        for i in range(self.d):
            integrand = lambda x: func(x)*self.basis(i, x)*self.weight(x)
            b[i], _ = quad(integrand, self.params[i]['l'], self.params[i]['r'], limit = INTLIM_PER_PIECE)
        # coeff = solve(self.G, b)
        coeff = pinv(self.G)@b
        # print("b =", b)
        # print("c =", coeff)
        
        def approx(x):
            ret = 0
            for i, c in enumerate(coeff):
                ret += c*self.basis(i, x)
            return ret

        if method == None:
            noise = np.zeros(self.d)
        elif method == 'Laplace':
            noise = np.random.normal(0, 1, (self.d,))
            noise = np.sqrt(np.random.exponential(1))*noise
            noise = (sqrtm(pinv(self.G))@noise).real
            noise = noise/eps
        elif method == 'Normal':
            noise = np.random.normal(0, 1, (self.d,))
            noise = (sqrtm(pinv(self.G))@noise).real/np.sqrt(2)
            noise = noise/eps
        else:
            print("ERR: no such method '{method}'.")

        def priv_approx(x):
            ret = 0
            for i in range(self.d):
                ret += (coeff[i]+noise[i])*self.basis(i, x)
            return ret

        return coeff, noise, approx, priv_approx
    
    def eval(self, func, approx, priv_approx):
        integrand = lambda x: (func(x)-approx(x))**2
        err_ls, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        err_ls = np.sqrt(err_ls)
        integrand = lambda x: (func(x)-priv_approx(x))**2
        err_priv, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        err_priv = np.sqrt(err_priv)
        return err_ls, err_priv

def l2_dist(f, g, l, r):
    integrand = lambda x: (f(x)-g(x))**2
    dist, _ = quad(integrand, l, r, limit = INTLIM)
    dist = np.sqrt(dist)
    return dist

def reduce_seg(func, interval, degree, k0, k, total_eps, eps, beta):
    if k == 0:
        return np.array(interval), total_eps
    breakpoints = np.linspace(interval[0], interval[1], (2**k)+1)
    solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
    _, _, approx, _ = solver.solve(func, 0, method = None)
    err = l2_dist(func, approx, interval[0], interval[1])
    noise = laplace.rvs(scale = 1/eps)
    total_eps = total_eps-eps
    print(f"ReduceSeg at interval {interval} with breakpoints {breakpoints}:\n", 
          f"\tB = {total_eps:.5f}, eps = {eps:.5f}, err = {err:.5f}, noise = {noise:.5f}, offset = {log(2)*k0, log(1/beta)}, {(log(2)*k0-log(1/beta))/eps:.5f};", 
          f"Proceed? {err+noise+(np.log(2)*k0-np.log(1/beta))/eps <= 2**(k-1)}.")
    
    if err+noise+(np.log(2)*k0-np.log(1/beta))/eps <= 2**(k-1):
        if k == 1:
            breakpoints = np.array(interval)
        else:
            l_pts, total_eps = reduce_seg(func, (interval[0], (interval[0]+interval[1])/2), degree, k0, k-2, total_eps, eps/4, beta)
            r_pts, total_eps = reduce_seg(func, ((interval[0]+interval[1])/2, interval[1]), degree, k0, k-2, total_eps, eps/4, beta)
            breakpoints = np.concatenate((l_pts[:-1], r_pts))
    return breakpoints, total_eps

def adaptive_poly_approx(func, interval, degree, eps = 0.5, beta = 0.5, method = 'Laplace'):
    print("="*100)
    print(f"Adaptive Private Function Approximation with degree-{degree} polynomials (eps = {eps}, beta = {beta})")
    # SVT using eps/4 privacy quota (eps_1 = eps/12, eps_2 = eps/6)
    k_bar = 0
    w = laplace.rvs(scale = 12/eps)
    while True:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
        solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
        _, _, approx, _ = solver.solve(func, 0, method = None)
        v = laplace.rvs(scale = 12/eps)
        tau = (2**k_bar)*4/eps
        err = l2_dist(func, approx, interval[0], interval[1])
        print(f"SVT: k_bar = {k_bar}, tau = {tau}, err = {err:.5f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
        if tau-err+v >= w:
            break
        k_bar = k_bar+1
    print(f"Final k_bar = {k_bar}.")
    print("-"*100)

    B = 3*eps/4
    k0 = min(k_bar-2, 4)
    if k0 < 1:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
    else:
        # recursively merge pieces in subintervals
        # 3/4*eps privacy quota left, each quad-interval consumes 1/16*eps quota
        # then there will be at least 1/2*eps quota left for final privatization
        breakpoints = np.array([interval[0]])
        for i in range(4):
            pts, B = reduce_seg(func, (interval[0]+(interval[1]-interval[0])*i/4, interval[0]+(interval[1]-interval[0])*(i+1)/4), 
                                degree, k0, k_bar-2, B, eps/32, beta)
            breakpoints = np.concatenate((breakpoints[:-1], pts))
    print(f"Final breakpoints: {breakpoints}, remaining eps = {B:.5f}.")
    print("-"*100)

    solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
    _, _, approx, priv = solver.solve(func, B, method = method)
    err_ls = l2_dist(func, approx, interval[0], interval[1])
    err_priv = l2_dist(func, priv, interval[0], interval[1])
    print(f"Polynomial approximation (eps = {B:.5f}, degree {solver.degree}):")
    print(f"\tbreakpoints: {solver.breakpoints};")
    print(f"\t||f-f_approx|| = {err_ls:.5f}, ||f-f_priv|| = {err_priv:.5f}.\n")
    return approx, priv, err_ls, err_priv

def plot_1D_1D(func, interval, approx, priv, err_ls, err_priv, eps, degree, method):
    x_plot = np.linspace(interval[0], interval[1], INTLIM)
    y_true = func(x_plot)
    y_approx = approx(x_plot)
    y_priv = priv(x_plot)

    plt.figure(figsize = (16, 10))
    plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'True Function')
    plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Poly Approx')
    plt.plot(x_plot, y_priv, 'b--', linewidth = 1.5, label = 'Private Poly Approx ({method})')
    plt.title(f"eps = {eps}, degree = {degree}, error: {err_priv:.5f} ({err_ls:.5f})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def time_series_func(t, val):
    def func(x):
        if isinstance(x, (int, float)):
            for i in range(len(t)-1):
                if x >= t[i] and x <= t[i+1]:
                    return (t[i+1]-x)/(t[i+1]-t[i])*val[i]+(x-t[i])/(t[i+1]-t[i])*val[i+1]
            return 0
        else:
            return np.array([func(y) for y in x])
    return func

if __name__ == "__main__":
    # ignore integration precision warnings
    warnings.simplefilter("ignore", category = IntegrationWarning)

    func = lambda x: 100*np.exp(-(x-20)**2/(2*8*8))+200*np.exp(-(x-80)**2/(2*5*5))
    approx, priv, err_ls, err_priv = adaptive_poly_approx(func = func, interval = (0, 100), degree = 3)
    plot_1D_1D(func, (0, 100), approx, priv, err_ls, err_priv, 0.5, 3, 'Laplace')

    """
    func = lambda x: 10*np.exp(-(x-2)**2/(2*0.8*0.8))+20*np.exp(-(x-8)**2/(2*0.5*0.5))
    approx, priv, err_ls, err_priv = adaptive_poly_approx(func = func, interval = (0, 10), degree = 3)
    plot_1D_1D(func, (0, 10), approx, priv, err_ls, err_priv, 0.5, 3, 'Laplace')
    """

    """
    t = np.linspace(0, 100, 1001)
    val = np.zeros(1001)
    for i in range(1001):
        val[i] = val[i]+100*np.exp(-(t[i]-20)**2/(2*8*8))
        val[i] = val[i]+200*np.exp(-(t[i]-80)**2/(2*5*5))
    approx, priv, err_ls, err_priv = adaptive_poly_approx(func = time_series_func(t, val), interval = (0, 100), degree = 3)
    plot_1D_1D(time_series_func(t, val), (0, 100), approx, priv, err_ls, err_priv, 0.5, 3, 'Laplace')
    """

    """
    t = np.linspace(0, 100, 101)
    val = np.random.rand(101)
    for i in range(101):
        val[i] = val[i]+100*np.exp(-(t[i]-40)**2/(2*5*5))
        val[i] = val[i]+100*np.exp(-(t[i]-60)**2/(2*10*10))
        val[i] = val[i]+25*np.exp(-(t[i]-20)**2/(2*8*8))
        val[i] = val[i]+40*np.exp(-(t[i]-80)**2/(2*5*5))
    adaptive_poly_approx(func = time_series_func(t, val), interval = (0, 100), degree = 12)
    """

    func = lambda x: 10*np.sin(0.02*np.pi*x)+5*np.cos(0.04*np.pi*x)
    approx, priv, err_ls, err_priv = adaptive_poly_approx(func = func, interval = (-100, 100), degree = 10, eps = 1)
    plot_1D_1D(func, (-100, 100), approx, priv, err_ls, err_priv, 1, 10, 'Laplace')
