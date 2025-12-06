from PrivPwcApprox import *

logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.INFO)

def reduce_seg(func, interval, basis, degree, k0, k, total_eps, eps, beta, 
               method = 'Laplace', time_series = None, parallel = False):
    if k <= 0: return np.array(interval), total_eps
    breakpoints = np.linspace(interval[0], interval[1], (2**k)+1)
    solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel = parallel)
    solver.fit(func, time_series, parallel)
    err = solver.eval('Approx')
    thresh = (2**(k-1))*solver.d
    if basis == 'Linear-2D': thresh *= 2
    if method == 'Laplace':
        noise = laplace.rvs(scale = 1/eps)
        thresh = (thresh*np.e)/(2*(np.e-1)*total_eps)
        thresh -= 1/np.sqrt(2*eps) * (np.log(2)*k0-np.log(1/beta))
    elif method == 'Gaussian':
        noise = norm.rvs(scale = 1/np.sqrt(2*eps))
        thresh = (np.sqrt(2)-1)/np.sqrt(2) * np.sqrt(thresh/(2*total_eps))
        thresh -= np.sqrt(2/eps) * (np.log(2)*k0-np.log(1/beta))
    total_eps = total_eps-eps
    logging.info(f"ReduceSeg at interval {interval}:"+
                 f"B = {total_eps:.5f}, eps = {eps:.8f}, err = {err:.5f}, noise = {noise:.5f}, "+
                 f"threshold = {thresh:.5f}; "+
                 f"Proceed? {err+noise <= thresh}.")
    
    if err+noise <= thresh:
        if k == 1:
            breakpoints = np.array(interval)
        else:
            l_pts, total_eps = reduce_seg(func, (interval[0], (interval[0]+interval[1])/2), basis, degree, 
                                          k0, k-2, total_eps, eps/4, beta, method, time_series, parallel)
            r_pts, total_eps = reduce_seg(func, ((interval[0]+interval[1])/2, interval[1]), basis, degree, 
                                          k0, k-2, total_eps, eps/4, beta, method, time_series, parallel)
            breakpoints = np.concatenate((l_pts[:-1], r_pts))
    return breakpoints, total_eps

def adaptive_approx(func, interval, basis = 'Polynomial', degree = 1, 
                    eps = 0.01, beta = 0.1, method = 'Laplace', SVT_threshold_scale = 1, 
                    time_series = None, parallel = False, enable_reduce_seg = True):
    logging.info("="*100)
    logging.info(f"Adaptive Private Function Approximation with degree-{degree} polynomials (eps = {eps}, beta = {beta})")

    # SVT using eps/4 privacy quota (eps_1 = eps/12, eps_2 = eps/6)
    # For CGP, eps_0 = sqrt(eps/2), eps_1 = eps_0/3, eps_2 = 2*eps_0/3
    k_bar = 0
    if method == 'Laplace': eps0 = eps/4
    elif method == 'Gaussian': eps0 = np.sqrt(eps/2)
    w = laplace.rvs(scale = 3/eps0)
    while True:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
        solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel = parallel)
        solver.fit(func, time_series, parallel)
        v = laplace.rvs(scale = 3/eps0)
        tau = (2**k_bar)*solver.d
        if basis == 'Linear-2D': tau *= 2
        if method == 'Gaussian': tau = np.sqrt(tau)
        tau = SVT_threshold_scale*tau/eps0
        err = solver.eval('Approx')
        logging.info(f"SVT: k_bar = {k_bar}, tau = {tau}, err = {err:.8f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
        if tau-err+v >= w: break
        k_bar = k_bar+1
    logging.info(f"Final k_bar = {k_bar}.")
    logging.info("-"*100)

    B = 3*eps/4
    if enable_reduce_seg:
        k0 = min(k_bar-2, 4)
        if k0 >= 1:
            # recursively merge pieces in subintervals
            # 3/4*eps privacy quota left, each quad-interval consumes 1/16*eps quota
            # then there will be at least 1/2*eps quota left for final privatization
            breakpoints = np.array([interval[0]])
            for i in range(4):
                pts, B = reduce_seg(func, (interval[0]+(interval[1]-interval[0])*i/4, interval[0]+(interval[1]-interval[0])*(i+1)/4), 
                                    basis, degree, k0, k_bar-2, B, eps/32, beta, method, time_series, parallel)
                breakpoints = np.concatenate((breakpoints[:-1], pts))
            solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel = parallel)
            solver.fit(func, time_series, parallel)
    logging.info(f"Final breakpoints: {breakpoints}, remaining eps = {B:.8f}.")

    return solver, B
