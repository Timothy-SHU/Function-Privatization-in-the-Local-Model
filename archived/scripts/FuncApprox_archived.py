import numpy as np
import matplotlib.pyplot as plt
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
        """
        numpts = int(self.r-self.l)*1000
        f = func(np.linspace(self.l, self.r, numpts))
        f_hat = approx(np.linspace(self.l, self.r, numpts))
        f_tilde = priv_approx(np.linspace(self.l, self.r, numpts))
        err_ls = np.sqrt(np.sum((f-f_hat)**2)/numpts)
        err_priv = np.sqrt(np.sum((f-f_tilde)**2)/numpts)
        """
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

def poly_approx(func, interval, breakpoints, degree, eps = 0.5, weight = None):
    plt.figure(figsize = (16, 5*((len(degree)-1)//2+1)))
    for i in range(len(degree)):
        solver = PrivateLeastSquaresApprox(interval, breakpoints[i], degree[i], weight)
        _, _, approx, priv_lap = solver.solve(func, eps, method = 'Laplace')
        _, _, approx, priv_norm = solver.solve(func, eps, method = 'Normal')
        err_ls = l2_dist(func, approx, interval[0], interval[1])
        err_lap = l2_dist(func, priv_lap, interval[0], interval[1])
        err_norm = l2_dist(func, priv_norm, interval[0], interval[1])
        # err_ls, err_lap = solver.eval(target_func, approx, priv_lap)
        # err_ls, err_norm = solver.eval(target_func, approx, priv_norm)
        print(f"Polynomial approximation (degree {solver.degree}, breakpoints {solver.breakpoints}) error: {err_lap:.5f} {err_norm:.5f} ({err_ls:.5f})")
    
        x_plot = np.linspace(interval[0], interval[1], INTLIM)
        y_true = func(x_plot)
        y_approx = approx(x_plot)
        y_priv_lap = priv_lap(x_plot)
        y_priv_norm = priv_norm(x_plot)
    
        plt.subplot((len(degree)-1)//2+1, 2, i+1)
        plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'True Function')
        plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Poly Approx')
        plt.plot(x_plot, y_priv_lap, 'b--', linewidth = 1.5, label = 'Private Poly Approx (Laplace)')
        plt.plot(x_plot, y_priv_norm, 'g--', linewidth = 1.5, label = 'Private Poly Approx (Normal)')
        plt.title(f"eps = {eps}, degree = {solver.degree}, {len(solver.breakpoints)-1} pieces, error: {err_lap:.5f} (GP), {err_norm:.5f} (CGP), {err_ls:.5f} (approx)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

def reduce_seg(func, interval, degree, k_bar, k, total_eps, eps, beta):
    if k == 0:
        return np.array(interval), total_eps
    breakpoints = np.linspace(interval[0], interval[1], (2**k)+1)
    solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
    _, _, approx, _ = solver.solve(func, 0, method = None)
    err = l2_dist(func, approx, interval[0], interval[1])
    noise = laplace.rvs(scale = 1/eps)
    total_eps = total_eps-eps
    print(f"ReduceSeg at interval {interval} with breakpoints {breakpoints}:\n", 
          f"\tB = {total_eps:.5f}, eps = {eps:.5f}, err = {err:.5f}, noise = {noise:.5f}, offset = {np.log2((2*k_bar)/(4*beta))/eps:.5f};", 
          f"Proceed? {err+noise+np.log2((2*k_bar)/(4*beta))/eps <= 2**k}.")
          # f"Proceed? {err+noise <= 2**k}.")
    
    if err+noise+np.log2((2*k_bar)/(4*beta))/eps <= 2**k:
    # if err+noise <= 2**k:
        if k == 1:
            breakpoints = np.array(interval)
        else:
            l_pts, total_eps = reduce_seg(func, (interval[0], (interval[0]+interval[1])/2), degree, k_bar, k-2, total_eps, eps, beta)
            r_pts, total_eps = reduce_seg(func, ((interval[0]+interval[1])/2, interval[1]), degree, k_bar, k-2, total_eps, eps, beta)
            breakpoints = np.concatenate((l_pts[:-1], r_pts))
    return breakpoints, total_eps

def adaptive_poly_approx(func, interval, degree, eps = 0.5, beta = 0.5):
    print("="*100)
    print(f"Adaptive Private Function Approximation with degree-{degree} polynomials (eps = {eps}, beta = {beta})")
    # SVT using eps/4 privacy quota (eps_1 = eps_2 = eps/8)
    k_bar = 0
    w = laplace.rvs(scale = 8/eps)
    while True:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
        solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
        _, _, approx, _ = solver.solve(func, 0, method = None)
        v = laplace.rvs(scale = 16/eps)
        tau = (2**k_bar)*4/eps
        err = l2_dist(func, approx, interval[0], interval[1])
        print(f"SVT: k_bar = {k_bar}, tau = {tau}, err = {err:.5f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
        if tau-err+v >= w:
            break
        k_bar = k_bar+1
    print(f"Final k_bar = {k_bar}.")
    print("-"*100)

    B = 3*eps/4
    if k_bar == 0:
        breakpoints = np.array(interval)
    else:
        # recursively merge pieces in left & right intervals
        # in the worst case, we need ??? iterations in total
        # 3/4*eps privacy quota left, let each iteration consume eps/(2*k_bar)
        # then there will be at least ??? quota left for final privatization

        # breakpoints, B = reduce_seg(func, interval, degree, k_bar, k_bar, B, eps/(4*k_bar), beta)
        l_pts, B = reduce_seg(func, (interval[0], (interval[0]+interval[1])/2), degree, k_bar, k_bar-1, B, eps/(2*k_bar), beta)
        r_pts, B = reduce_seg(func, ((interval[0]+interval[1])/2, interval[1]), degree, k_bar, k_bar-1, B, eps/(2*k_bar), beta)
        breakpoints = np.concatenate((l_pts[:-1], r_pts))
    print(f"Final breakpoints: {breakpoints}, remaining eps = {B:.5f}.")
    print("-"*100)

    solver = PrivateLeastSquaresApprox(interval, breakpoints, degree)
    _, _, approx, priv = solver.solve(func, B, method = 'Laplace')
    err_ls = l2_dist(func, approx, interval[0], interval[1])
    err_priv = l2_dist(func, priv, interval[0], interval[1])
    print(f"Polynomial approximation (eps = {B:.5f}, degree {solver.degree}):")
    print(f"\tbreakpoints: {solver.breakpoints};")
    print(f"\t||f-f_approx|| = {err_ls:.5f}, ||f-f_priv|| = {err_priv:.5f}.\n")

    x_plot = np.linspace(interval[0], interval[1], INTLIM)
    y_true = func(x_plot)
    y_approx = approx(x_plot)
    y_priv = priv(x_plot)

    plt.figure(figsize = (16, 10))
    plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'True Function')
    plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Poly Approx')
    plt.plot(x_plot, y_priv, 'b--', linewidth = 1.5, label = 'Private Poly Approx (Laplace)')
    plt.title(f"eps = {eps}, degree = {solver.degree}, breakpoints = {breakpoints}, error: {err_priv:.5f} ({err_ls:.5f})")
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

def public_breakpoints_examples():
    CASE = int(input())
    if CASE == 1:
        poly_approx(func = lambda x: 10*np.sin(2*np.pi*x)+5*np.cos(4*np.pi*x),
                    interval = (-1, 1), 
                    breakpoints = [ [-1, 1], 
                                    [-1, 0, 1], 
                                    [-1, -0.5, 0, 0.5, 1],
                                    [-1, -0.9, -0.75, -0.6, -0.25, 0.1, 0.25, 0.4, 0.75, 1] ], 
                    degree = [ 12, 6, 3, 1 ], eps = 1)
    elif CASE == 2:
        t = np.linspace(0, 10, 101)
        val = np.random.rand(101)
        for i in range(101):
            val[i] = val[i]+np.exp(t[i]/2)+np.sin(i)
        poly_approx(func = time_series_func(t, val), 
                    interval = (0, 10), 
                    breakpoints = [ [0, 2.5, 5, 7.5, 10],
                                    [0, 2.5, 5, 7.5, 10],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ], 
                    degree = [ 1, 2, 1, 2 ])
    elif CASE == 3:
        t = np.linspace(0, 100, 101)
        val = np.random.rand(101)
        for i in range(101):
            val[i] = val[i]+100*np.exp(-(t[i]-40)**2/(2*5*5))
            val[i] = val[i]+100*np.exp(-(t[i]-60)**2/(2*10*10))
            val[i] = val[i]+25*np.exp(-(t[i]-20)**2/(2*8*8))
            val[i] = val[i]+40*np.exp(-(t[i]-80)**2/(2*5*5))
        poly_approx(func = time_series_func(t, val), 
                    interval = (0, 100), 
                    breakpoints = [ [0, 100], 
                                    [0, 50, 100], 
                                    np.linspace(0, 100, 6), 
                                    np.linspace(0, 100, 11) ], 
                    degree = [ 12, 6, 4, 1 ])
    elif CASE == 4:
        t = np.linspace(0, 100, 101)
        val = np.random.rand(101)
        for i in range(101):
            val[i] = val[i]+100*np.exp(-(t[i]-40)**2/(2*5*5))
            val[i] = val[i]+100*np.exp(-(t[i]-60)**2/(2*10*10))
            val[i] = val[i]+25*np.exp(-(t[i]-20)**2/(2*8*8))
            val[i] = val[i]+40*np.exp(-(t[i]-80)**2/(2*5*5))
        poly_approx(func = time_series_func(t, val), 
                    interval = (0, 100), 
                    breakpoints = [ np.linspace(0, 100, 51), 
                                    np.linspace(0, 100, 51) ], 
                    degree = [ 1, 2 ])

if __name__ == "__main__":
    # ignore integration precision warnings
    warnings.simplefilter("ignore", category = IntegrationWarning)

    # public_breakpoints_examples()

    t = np.linspace(0, 100, 101)
    val = np.zeros(101)
    for i in range(101):
        val[i] = val[i]+100*np.exp(-(t[i]-20)**2/(2*8*8))
    adaptive_poly_approx(func = time_series_func(t, val), interval = (0, 100), degree = 3)

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

    # adaptive_poly_approx(func = lambda x: 10*np.sin(2*np.pi*x)+5*np.cos(4*np.pi*x), interval = (-1, 1), degree = 12, eps = 1)
