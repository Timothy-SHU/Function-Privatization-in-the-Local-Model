from PrivPwcApprox import *

logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.INFO)

def adaptive_basis(func, interval, basis = 'Polynomial', eps = 0.1, method = 'Laplace', 
                   enable_filtering = True, STEP_SIZE = 4, MAX_DEGREE = 24):
    # SVT for basis degree, uses 1/4 budget
    k0 = 0; solver = None
    if method == 'Laplace': eps0 = eps/4
    elif method == 'Gaussian': eps0 = np.sqrt(eps/2)
    w = laplace.rvs(scale = 3/eps0)
    while True:
        solver = PrivatePiecewiseApprox(interval, list(interval), basis, k0)
        solver.fit(func)
        if k0 >= MAX_DEGREE: break
        v = laplace.rvs(scale = 3/eps0)
        # if method == 'Laplace': tau = k0/eps0
        # elif method == 'Gaussian': tau = np.sqrt(k0)/eps0
        if enable_filtering: tau = 1/eps0
        elif method == 'Laplace': tau = k0/(3*eps/4)
        elif method == 'Gaussian': tau = np.sqrt(k0)/np.sqrt(3*eps/2)
        err = solver.eval('Approx')
        logging.info(f"SVT: k0 = {k0}, tau = {tau}, err = {err:.8f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
        if tau-err+v >= w: break
        k0 = k0+STEP_SIZE
    logging.info(f"Final k0 = {k0}.")
    logging.info("-"*100)

    if enable_filtering:
        # Privatize with 1/4 budget
        solver.privatize(eps/4, method)
        cmax = np.abs(solver.coeff+solver.noise).max()
        c_list = sorted(np.abs(solver.coeff+solver.noise).ravel().tolist(), reverse = True)

        # SVT for basis selection, uses 1/4 budget
        k1 = 0; deg_list = None
        if method == 'Laplace': eps1 = eps/4
        elif method == 'Gaussian': eps1 = np.sqrt(eps/2)
        w = laplace.rvs(scale = 3/eps1)
        original_noise = solver.noise.copy()
        while True:
            deg_list = []
            for i in range(k0+1):
                ## Instead of using threshold 2^k1, simply use the k1-th largest coeff
                # if np.abs(solver.coeff[0, i]+solver.noise[0, i]) >= cmax/(2**k1): deg_list.append(i)
                if np.abs(solver.coeff[0, i]+solver.noise[0, i]) >= c_list[k1]: deg_list.append(i)
                else: solver.noise[0, i] = -solver.coeff[0, i]
            if len(deg_list) == k0+1: break
            v = laplace.rvs(scale = 3/eps1)
            if method == 'Laplace': tau = len(deg_list)/eps0
            elif method == 'Gaussian': tau = np.sqrt(len(deg_list))/eps0
            err = solver.eval('Priv')
            solver.noise = original_noise.copy()
            # logging.info(f"SVT: k1 = {k1} ({cmax/(2**k1)}), tau = {tau}, err = {err:.8f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
            logging.info(f"SVT: k1 = {k1} ({c_list[k1]}), tau = {tau}, err = {err:.8f}, w = {w:.5f}, v = {v:.5f}; Terminate? {tau-err+v >= w}.")
            if tau-err+v >= w: break
            k1 = k1+1
        logging.info(f"Final k1 = {k1}.")
        logging.info("-"*100)
    else: deg_list = list(range(k0+1))

    # Final privatization with 1/4 budget
    logging.info(f"Final degree list: {deg_list}, remaining eps = {eps/4:.8f}")
    solver = PrivatePiecewiseApprox(interval, list(interval), basis, deg_list)
    solver.fit(func)
    if enable_filtering: solver.privatize(eps/4, method)
    else: solver.privatize(3*eps/4, method)

    return solver, deg_list