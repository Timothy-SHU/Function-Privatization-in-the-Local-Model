from PrivPwcApprox_quad import *

logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.INFO)

def reduce_seg(func, interval, basis, degree, 
               k0, k, total_eps, eps, beta, 
               func_2D = None, parallel = False):
    if k == 0:
        return np.array(interval), total_eps
    breakpoints = np.linspace(interval[0], interval[1], (2**k)+1)
    solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel)
    solver.fit(func, func_2D, parallel)
    err = solver.eval('Approx')
    noise = laplace.rvs(scale = 1/eps)
    total_eps = total_eps-eps
    logging.info(f"ReduceSeg at interval {interval} with breakpoints {breakpoints}:\n"+
                 f"\tB = {total_eps:.5f}, eps = {eps:.5f}, err = {err:.5f}, noise = {noise:.5f}, "+
                 f"offset = {(log(2)*k0-log(1/beta))/eps:.5f}; "+
                 f"Proceed? {err+noise+(np.log(2)*k0-np.log(1/beta))/eps <= 2**(k-1)}.")
    
    if err+noise+(np.log(2)*k0-np.log(1/beta))/eps <= 2**(k-1):
        if k == 1:
            breakpoints = np.array(interval)
        else:
            l_pts, total_eps = reduce_seg(func, (interval[0], (interval[0]+interval[1])/2), basis, degree, 
                                          k0, k-2, total_eps, eps/4, beta, func_2D, parallel)
            r_pts, total_eps = reduce_seg(func, ((interval[0]+interval[1])/2, interval[1]), basis, degree, 
                                          k0, k-2, total_eps, eps/4, beta, func_2D, parallel)
            breakpoints = np.concatenate((l_pts[:-1], r_pts))
    return breakpoints, total_eps

def adaptive_approx(func, interval, basis = 'Polynomial', degree = 1, 
                    eps = 1, beta = 0.1, method = 'Laplace', 
                    func_2D = None, SVT_threshold_scale = 1, parallel = False):
    logging.info("="*100)
    logging.info(f"Adaptive Private Function Approximation with degree-{degree} polynomials (eps = {eps}, beta = {beta})")
    # SVT using eps/4 privacy quota (eps_1 = eps/12, eps_2 = eps/6)
    k_bar = 0
    w = laplace.rvs(scale = 12/eps)
    while True:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
        solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel)
        solver.fit(func, func_2D, parallel)
        v = laplace.rvs(scale = 12/eps)
        tau = (2**k_bar)*4/eps
        err = solver.eval()
        logging.info(f"SVT: k_bar = {k_bar}, tau = {tau}, err = {err:.5f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
        if tau-(err/SVT_threshold_scale)+v >= w:
            break
        k_bar = k_bar+1
    logging.info(f"Final k_bar = {k_bar}.")
    logging.info("-"*100)

    B = 3*eps/4
    """
    k0 = min(k_bar-2, 4)
    if k0 == 1:
        breakpoints = np.linspace(interval[0], interval[1], (2**k_bar)+1)
    else:
        # recursively merge pieces in subintervals
        # 3/4*eps privacy quota left, each quad-interval consumes 1/16*eps quota
        # then there will be at least 1/2*eps quota left for final privatization
        breakpoints = np.array([interval[0]])
        for i in range(4):
            pts, B = reduce_seg(func, (interval[0]+(interval[1]-interval[0])*i/4, interval[0]+(interval[1]-interval[0])*(i+1)/4), 
                                basis, degree, k0, k_bar-2, B, eps/32, beta, func_2D = func_2D)
            breakpoints = np.concatenate((breakpoints[:-1], pts))
    logging.info(f"Final breakpoints: {breakpoints}, remaining eps = {B:.5f}.")
    logging.info("-"*100)

    solver = PrivatePiecewiseApprox(interval, breakpoints, basis, degree, parallel)
    solver.fit(func, func_2D, parallel)
    """
    solver.privatize(B, method)
    logging.info(f"Polynomial approximation (eps = {B:.5f}, degree {solver.degree}):")
    logging.info(f"\tbreakpoints: {solver.breakpoints}.")
    # logging.info(f"\t||f-f_approx||: {solver.eval('Approx')}; ||f-f_priv||: {solver.eval('Priv')}.")
    return solver
