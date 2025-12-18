import sys, random, tqdm
from PrivPwcApprox import *
from AdaptApprox import *

repeat = 10
SAVE_FIGS = True

def genNoise(method, scale):
    if method == 'Laplace':
        return laplace.rvs(scale = scale)
    elif method == 'Gaussian':
        return norm.rvs(scale = scale)
    return None

def genGaus(params):
    def func(x):
        ret = 0
        for scale, center, sd in params:
            ret += scale*np.exp(-(x-center)**2/(2*sd*sd))
        return ret
    return func

def genTrig(params):
    def func(x):
        ret = 0
        for scale, type, k in params:
            if type == 's': ret += scale*np.sin(k*(x-50)/50*np.pi)
            elif type == 'c': ret += scale*np.cos(k*(x-50)/50*np.pi)
        return ret
    return func

FUNC_LIST = [genGaus([(100, 50, 10)]), 
             genGaus([(100, 20, 8), (200, 80, 5)]), 
             genGaus([(100, 20, 4), (200, 50, 5), (400, 80, 4)]), 
             genGaus([(50, 5, 2), (200, 25, 8), (150, 50, 10), (400, 75, 4), (100, 90, 8)]), 
             genGaus([(100, 30, 8), (-200, 50, 10), (200, 80, 5)])]
DEGREE_LIST = [1, 4, 8, 16]
SAMPLE_LIST = [10, 20, 50, 100]
WINDOW_SCALE = 0.1
EPS_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
RHO_LIST = [1e-6, 4e-6, 2.5e-5, 1e-4, 4e-4, 2.5e-3, 0.01]

# SAMPLE_LIST = [10, 20]
# EPS_LIST = [0.01, 0.1, 1.0]
# RHO_LIST = [1e-6, 1e-4, 0.01]

def genRandomFunc(n, SEED = 42):
    global FUNC_LIST
    FUNC_LIST = []; random.seed(SEED)
    for i in range(n):
        params = []; m = random.randint(1, 10)
        for j in range(m):
            params.append((random.randint(50, 500), random.randint(0, 100), random.randint(2, 16)))
        # print(params)
        FUNC_LIST.append(genGaus(params))

def plotEg(idx, method, eps = 0.1, SAMPLE_LIST = [10, 20]):
    func = FUNC_LIST[idx]
    if method == 'Gaussian': eps = eps*eps/100
    print("="*72+f" Curve #{idx+1:2d} "+"="*72)
    
    plt.figure(figsize = (15, 9))
    for idx, DEGREE in enumerate(DEGREE_LIST):
        solver, B = adaptive_approx(func = func, interval = (0, 100), 
                                    basis = 'Polynomial', degree = DEGREE, 
                                    eps = eps, method = method)
        solver.privatize(eps = B, method = method)
        print(f"Polynomial Basis of degree {DEGREE}:", end = "\t")
        print(f"||f|| = {np.sqrt(solver.funcSqrInt):.5f};", end = "\t")
        print(f"||f-f_approx|| = {solver.eval('Approx'):.5f};", end = "\t")
        # print(f"||f_approx-f_priv|| = {solver.evalPrivLoss():.5f};", end = "\t")
        print(f"||f-f_priv|| = {solver.eval('Priv'):.5f};", end = "\t")
        plt.subplot(2, 3, idx//2*3+idx%2+1)
        x_dense = np.linspace(0, 100, INTLIM)
        plt.plot(x_dense, func(x_dense), color = 'black', label = "Function")
        for k in range(len(solver.breakpoints)-1):
            l = solver.breakpoints[k]; r = solver.breakpoints[k+1]
            x_piece = np.linspace(l, r, INTLIM_PER_PIECE)[:-1]
            plt.plot(x_piece, solver.createApprox()(x_piece), color = 'tab:brown', 
                     alpha = 0.6, label = "LS Approximation" if k == 0 else None)
            plt.plot(x_piece, solver.createPriv()(x_piece), color = 'tab:blue', 
                     alpha = 0.6, label = "PrivFuncSeg" if k == 0 else None)
        solver.smooth()
        print(f"||f-f_smooth|| = {solver.eval('Priv'):.5f}.")
        plt.plot(x_dense, solver.createPriv()(x_dense), color = 'tab:orange', 
                 alpha = 0.9, label = "PrivFuncSeg (continuous)")
        plt.title(f"Degree-{DEGREE} Monomial Basis"); plt.legend(loc = 'upper left')

    for idx in range(2):
        plt.subplot(2, 3, idx*3+3)
        SAMPLE = SAMPLE_LIST[idx]
        WINDOW = max(1, int(SAMPLE*WINDOW_SCALE/2))
        sample = np.linspace(0, 100, SAMPLE)
        val_priv = func(sample)
        for i in range(SAMPLE):
            if method == 'Laplace':
                val_priv[i] += genNoise(method, SAMPLE/eps)
            elif method == 'Gaussian':
                val_priv[i] += genNoise(method, np.sqrt(SAMPLE/(2*eps)))
        val_smooth = np.zeros(SAMPLE)
        for l in range(SAMPLE):
            val_smooth[l] = np.mean(val_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
        integrand = lambda x: (func(x)-np.interp(x, sample, val_priv))**2
        print(f"Baseline with {SAMPLE} samples:", end = "\t")
        print(f"||f-f_bl|| = {np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0]):.5f};", end = "\t")
        integrand = lambda x: (func(x)-np.interp(x, sample, val_smooth))**2
        print(f"||f-f_bl_sm|| = {np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0]):.5f}.")
        plt.plot(x_dense, func(x_dense), color = 'black', label = "function")
        plt.plot(sample, val_priv, color = 'tab:green', alpha = 0.9, label = "Baseline")
        plt.plot(sample, val_smooth, color = 'tab:purple', alpha = 0.9, label = "Baseline (smoothed)")
        plt.title(f"Baseline with {SAMPLE} Samples"); plt.legend()

    plt.subplots_adjust(left = 0.03, right = 0.99, top = 0.96, bottom = 0.03, 
                        wspace = 0.13, hspace = 0.16)
    if SAVE_FIGS:
        filename = "results/figs/Synth_Eg_"
        filename += "GP_eps" if method == 'Laplace' else "CGP_rho"
        filename += f"={eps}.pdf"
        plt.savefig(filename)
    plt.show()

def expt(method):
    results = np.zeros((len(DEGREE_LIST), 5, len(EPS_LIST)))
    for j, EPS in enumerate(EPS_LIST):
        eps = EPS
        if method == 'Gaussian': eps = eps*eps/100
        print(f"Running {'GP' if method == 'Laplace' else 'CGP'} with eps = {EPS}...")
        pbar = tqdm.tqdm(total = len(FUNC_LIST)*repeat*(len(DEGREE_LIST)+len(SAMPLE_LIST)))
        for func in FUNC_LIST:
            funcL2 = None
            for i, DEGREE in enumerate(DEGREE_LIST):
                err_ls = 0; err_priv = 0; err_smooth = 0
                for k in range(repeat):
                    solver, B = adaptive_approx(func = func, interval = (0, 100), 
                                                basis = 'Polynomial', degree = DEGREE, 
                                                eps = EPS, method = method)
                    if funcL2 == None: funcL2 = np.sqrt(solver.funcSqrInt)
                    solver.privatize(eps = B, method = method)
                    err_ls += solver.eval('Approx')/funcL2
                    err_priv += solver.eval('Priv')/funcL2
                    solver.smooth()
                    err_smooth += solver.eval('Priv')/funcL2
                    pbar.update(1)
                err_ls /= repeat; err_priv /= repeat; err_smooth /= repeat
                # print(func, DEGREE, err_priv, err_smooth)
                results[i, 0, j] += err_ls/len(FUNC_LIST)
                results[i, 1, j] += err_priv/len(FUNC_LIST)
                results[i, 2, j] += err_smooth/len(FUNC_LIST)

            min_err_priv = None; min_err_smooth = None
            for SAMPLE in SAMPLE_LIST:
                WINDOW = max(1, int(SAMPLE*WINDOW_SCALE/2))
                err_priv = 0; err_smooth = 0
                for k in range(repeat):
                    sample = np.linspace(0, 100, SAMPLE)
                    val_priv = func(sample)
                    for i in range(SAMPLE):
                        if method == 'Laplace':
                            val_priv[i] += genNoise(method, SAMPLE/EPS)
                        elif method == 'Gaussian':
                            val_priv[i] += genNoise(method, np.sqrt(SAMPLE/(2*EPS)))
                    integrand = lambda x: (func(x)-np.interp(x, sample, val_priv))**2
                    err_priv += np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0])/funcL2
                    val_smooth = np.zeros(SAMPLE)
                    for l in range(SAMPLE):
                        val_smooth[l] = np.mean(val_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
                    integrand = lambda x: (func(x)-np.interp(x, sample, val_smooth))**2
                    err_smooth += np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0])/funcL2
                    pbar.update(1)
                err_priv /= repeat; err_smooth /= repeat
                # print(func, SAMPLE, err_priv)
                if min_err_priv == None: min_err_priv = err_priv
                else: min_err_priv = min(min_err_priv, err_priv)
                if min_err_smooth == None: min_err_smooth = err_smooth
                else: min_err_smooth = min(min_err_smooth, err_smooth)
            # print(func, min_err_priv, min_err_smooth)
            for i in range(len(DEGREE_LIST)):
                results[i, 3, j] += min_err_priv/len(FUNC_LIST)
                results[i, 4, j] += min_err_smooth/len(FUNC_LIST)
        pbar.close()

    names = []; colors = []; markers = []
    names.append("LS Approximation"); colors.append('tab:brown'); markers.append('P')
    names.append("PrivFuncSeg"); colors.append('tab:blue'); markers.append('^')
    names.append("PrivFuncSeg (continuous)"); colors.append('tab:orange'); markers.append('o')
    names.append("Baseline"); colors.append('tab:green'); markers.append('s')
    names.append("Baseline (smoothed)"); colors.append('tab:purple'); markers.append('d')

    budgets = EPS_LIST if method == 'Laplace' else RHO_LIST
    plt.figure(figsize = (12, 9))
    for i in range(len(DEGREE_LIST)):
        plt.subplot(2, 2, i+1)
        plt.axhline(y = 1, color = 'black', linestyle = '--')
        for j in range(1, len(names)):
            plt.plot(budgets, results[i, j, :].tolist(), color = colors[j], 
                    alpha = 0.9, marker = markers[j], label = names[j])
        plt.title(f"Degree-{DEGREE_LIST[i]} Monomial Basis")
        plt.legend(loc = 'upper right'); plt.xscale('log'); plt.yscale('log')
        plt.xticks(budgets, budgets, minor = False)
        if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
        elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
        plt.ylabel("Error")
    plt.subplots_adjust(left = 0.055, right = 0.99, top = 0.96, bottom = 0.06, 
                        wspace = 0.13, hspace = 0.23)
    if SAVE_FIGS:
        filename = "results/figs/Synth"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        plt.savefig(filename)
    else: plt.show()

# plotEg(1, 'Laplace')
genRandomFunc(30)
# for i in range(len(FUNC_LIST)):
#     plotEg(i, 'Gaussian', eps = 0.1, plotBaseline = True, SAMPLE = 20)
expt('Laplace')
expt('Gaussian')