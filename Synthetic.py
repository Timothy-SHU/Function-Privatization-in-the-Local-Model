import os, sys, random, tqdm
from PrivPwcApprox import *
from AdaptApprox import *

repeat = 50
SAVE_FIGS = True
GET_MSE = False

plt.rc('axes', titlesize = 12)
plt.rc('axes', labelsize = 12)
plt.rc('xtick', labelsize = 9)
plt.rc('ytick', labelsize = 9)
plt.rc('legend', fontsize = 10)

def getMSE(arr):
    avg = arr.mean()
    return ((arr-avg)**2).mean()

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

# EPS_LIST = [0.001, 0.01, 0.1, 1.0]
# RHO_LIST = [5e-7, 5e-5, 5e-3, 0.5]

EPS_LIST = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
RHO_LIST = [1e-5, 4e-5, 2.5e-4, 0.001, 0.004, 0.025, 0.1]
# RHO_LIST = [5e-5, 2e-4, 1.25e-3, 5e-3, 2e-2, 0.125, 0.5]

# EPS_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# RHO_LIST = [1e-7, 4e-7, 2.5e-6, 1e-5, 4e-5, 2.5e-4, 0.001, 0.004, 0.025, 0.1]
# RHO_LIST = [5e-7, 2e-6, 1.25e-5, 5e-5, 2e-4, 1.25e-3, 5e-3, 2e-2, 0.125, 0.5]

def genRandomFunc(n, SEED = 42):
    global FUNC_LIST
    FUNC_LIST = []; random.seed(SEED)
    for i in range(n):
        params = []; m = random.randint(1, 5)
        for j in range(m):
            params.append((random.randint(50, 500), random.randint(0, 100), random.randint(2, 16)))
        # print(params)
        FUNC_LIST.append(genGaus(params))

def plotEg(idx, method, eps = 0.1, SAMPLE_LIST = [10, 20]):
    func = FUNC_LIST[idx]
    if method == 'Gaussian': eps = eps*eps/100
    print("="*72+f" Curve #{idx+1:2d} "+"="*72)
    
    plt.figure(figsize = (8, 10))
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
        plt.subplot(3, 2, idx+1)
        x_dense = np.linspace(0, 100, INTLIM)
        plt.plot(x_dense, func(x_dense), color = 'black', label = "Function")
        for k in range(len(solver.breakpoints)-1):
            l = solver.breakpoints[k]; r = solver.breakpoints[k+1]
            x_piece = np.linspace(l, r, INTLIM_PER_PIECE)[:-1]
            # plt.plot(x_piece, solver.createApprox()(x_piece), color = 'tab:brown', 
            #          alpha = 0.6, label = "LS Approximation" if k == 0 else None)
            plt.plot(x_piece, solver.createPriv()(x_piece), color = 'tab:blue', 
                     alpha = 0.6, label = "PrivFuncSeg" if k == 0 else None)
        solver.smooth()
        print(f"||f-f_smooth|| = {solver.eval('Priv'):.5f}.")
        plt.plot(x_dense, solver.createPriv()(x_dense), color = 'tab:orange', 
                 alpha = 0.9, label = "PrivFuncSeg\n(continuous)")
        plt.title(f"Degree-{DEGREE} Monomial Basis"); plt.legend(loc = 'upper left')

    for idx in range(2):
        plt.subplot(3, 2, idx+5)
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
        plt.plot(x_dense, func(x_dense), color = 'black', label = "Function")
        plt.plot(sample, val_priv, color = 'tab:green', alpha = 0.9, label = "Baseline")
        plt.plot(sample, val_smooth, color = 'tab:purple', alpha = 0.9, label = "Baseline\n(smoothed)")
        plt.title(f"Baseline with {SAMPLE} Samples"); plt.legend()

    plt.subplots_adjust(left = 0.055, right = 0.99, top = 0.97, bottom = 0.025, 
                        wspace = 0.16, hspace = 0.2)
    if SAVE_FIGS:
        filename = "results/figs/Synth_eg_"
        filename += "GP_eps" if method == 'Laplace' else "CGP_rho"
        filename += f"={eps}.pdf"
        plt.savefig(filename)
    plt.show()

def expt(method):
    results = np.zeros((len(DEGREE_LIST), 10, len(EPS_LIST)))
    for j, EPS in enumerate(EPS_LIST):
        eps = EPS
        if method == 'Gaussian': eps = eps*eps/10
        filename = f"results/Synth/Synth_{EPS}_{method}.txt"
        if not os.path.isfile(filename):
            print(f"Running {'GP' if method == 'Laplace' else 'CGP'} with eps = {EPS}...")
            pbar = tqdm.tqdm(total = len(FUNC_LIST)*repeat*(len(DEGREE_LIST)+len(SAMPLE_LIST)))
            for func in FUNC_LIST:
                funcL2 = None
                for i, DEGREE in enumerate(DEGREE_LIST):
                    err_ls = []; err_priv = []; err_smooth = []
                    for k in range(repeat):
                        solver, B = adaptive_approx(func = func, interval = (0, 100), 
                                                    basis = 'Polynomial', degree = DEGREE, 
                                                    eps = EPS, method = method)
                        if funcL2 == None: funcL2 = np.sqrt(solver.funcSqrInt)
                        solver.privatize(eps = B, method = method)
                        err_ls.append(solver.eval('Approx')/funcL2)
                        err_priv.append(solver.eval('Priv')/funcL2)
                        solver.smooth()
                        err_smooth.append(solver.eval('Priv')/funcL2)
                        pbar.update(1)
                    err_ls = np.array(err_ls)
                    err_priv = np.array(err_priv)
                    err_smooth = np.array(err_smooth)
                    # print(func, DEGREE, err_priv, err_smooth)
                    results[i, 0, j] += err_ls.mean()/len(FUNC_LIST)
                    results[i, 1, j] += err_priv.mean()/len(FUNC_LIST)
                    results[i, 2, j] += err_smooth.mean()/len(FUNC_LIST)
                    results[i, 5, j] += getMSE(err_ls)/len(FUNC_LIST)
                    results[i, 6, j] += getMSE(err_priv)/len(FUNC_LIST)
                    results[i, 7, j] += getMSE(err_smooth)/len(FUNC_LIST)

                min_err_priv = None; min_err_smooth = None
                min_err_priv_MSE = None; min_err_smooth_MSE = None
                for SAMPLE in SAMPLE_LIST:
                    WINDOW = max(1, int(SAMPLE*WINDOW_SCALE/2))
                    err_priv = []; err_smooth = []
                    for k in range(repeat):
                        sample = np.linspace(0, 100, SAMPLE)
                        val_priv = func(sample)
                        for i in range(SAMPLE):
                            if method == 'Laplace':
                                val_priv[i] += genNoise(method, SAMPLE/EPS)
                            elif method == 'Gaussian':
                                val_priv[i] += genNoise(method, np.sqrt(SAMPLE/(2*EPS)))
                        integrand = lambda x: (func(x)-np.interp(x, sample, val_priv))**2
                        err_priv.append(np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0])/funcL2)
                        val_smooth = np.zeros(SAMPLE)
                        for l in range(SAMPLE):
                            val_smooth[l] = np.mean(val_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
                        integrand = lambda x: (func(x)-np.interp(x, sample, val_smooth))**2
                        err_smooth.append(np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0])/funcL2)
                        pbar.update(1)
                    err_priv = np.array(err_priv)
                    err_smooth = np.array(err_smooth)
                    # print(func, SAMPLE, err_priv)
                    if min_err_priv == None or err_priv.mean() < min_err_priv:
                        min_err_priv = err_priv.mean()
                        min_err_priv_MSE = getMSE(err_priv)
                    if min_err_smooth == None or err_smooth.mean() < min_err_smooth:
                        min_err_smooth = err_smooth.mean()
                        min_err_smooth_MSE = getMSE(err_smooth)
                # print(func, min_err_priv, min_err_smooth)
                for i in range(len(DEGREE_LIST)):
                    results[i, 3, j] += min_err_priv/len(FUNC_LIST)
                    results[i, 4, j] += min_err_smooth/len(FUNC_LIST)
                    results[i, 8, j] += min_err_priv_MSE/len(FUNC_LIST)
                    results[i, 9, j] += min_err_smooth_MSE/len(FUNC_LIST)
            os.makedirs("results/Synth/", exist_ok = True)
            res_file = open(filename, 'w')
            for i in range(len(DEGREE_LIST)):
                for k in range(10):
                    res_file.write(f"{results[i, k, j]} ")
                res_file.write("\n")
            res_file.close()
            pbar.close()
        else:
            print(f"Loading results for {'GP' if method == 'Laplace' else 'CGP'} with eps = {EPS}...", end = '\t')
            file = open(filename, 'r')
            buffer = [float(x) for x in file.read().strip().split()]
            for i in range(len(DEGREE_LIST)):
                for k in range(10):
                    results[i, k, j] = buffer[0]
                    buffer = buffer[1:]
            print("Complete.")
    
    if GET_MSE: results = results[:, 5:, :]
    else: results = results[:, :5, :]

    names = []; colors = []; markers = []
    names.append("LS Approximation"); colors.append('tab:brown'); markers.append('P')
    names.append("PrivFuncSeg"); colors.append('tab:blue'); markers.append('^')
    names.append("PrivFuncSeg (continuous)"); colors.append('tab:orange'); markers.append('o')
    names.append("Baseline"); colors.append('tab:green'); markers.append('s')
    names.append("Baseline (smoothed)"); colors.append('tab:purple'); markers.append('d')

    budgets = EPS_LIST if method == 'Laplace' else RHO_LIST
    _, axs = plt.subplots(2, 2, figsize = (8, 8), sharex = True, sharey = True)
    for i in range(len(DEGREE_LIST)):
        plt.subplot(2, 2, i+1)
        plt.axhline(y = 1, color = 'black', linestyle = '--')
        for j in range(1, len(names)):
            plt.plot(budgets, results[i, j, :].tolist(), color = colors[j], 
                    alpha = 0.9, marker = markers[j], label = names[j])
        plt.title(f"Degree-{DEGREE_LIST[i]} Monomial Basis")
        # plt.tick_params(labelleft = True)
        plt.tick_params(labelbottom = True)
        plt.legend(loc = 'upper right'); plt.xscale('log'); plt.yscale('log')
        plt.xticks(budgets, budgets, minor = False)
        if i//2 == 1:
            if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
            elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
        if i%2 == 0: plt.ylabel("Error MSE" if GET_MSE else "Error")
    plt.subplots_adjust(left = 0.08, right = 0.99, top = 0.97, bottom = 0.06, 
                        wspace = 0.04, hspace = 0.16)
    if SAVE_FIGS:
        filename = "results/figs/Synth"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        if GET_MSE: filename = filename[:-4]+"_MSE.pdf"
        plt.savefig(filename)
    
    _, axs = plt.subplots(2, 2, figsize = (8, 8), sharex = True, sharey = True)
    for i in range(len(DEGREE_LIST)):
        plt.subplot(2, 2, i+1)
        plt.axhline(y = 1, color = 'black', linestyle = '--')
        for j in range(len(names)):
            plt.plot(budgets, results[i, j, :].tolist(), color = colors[j], 
                    alpha = 0.9, marker = markers[j], label = names[j])
        plt.title(f"Degree-{DEGREE_LIST[i]} Monomial Basis")
        # plt.tick_params(labelleft = True)
        plt.tick_params(labelbottom = True)
        plt.legend(loc = 'upper right', fontsize = 9)
        plt.xscale('log'); plt.yscale('log')
        plt.xticks(budgets, budgets, minor = False)
        if i//2 == 1:
            if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
            elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
        if i%2 == 0: plt.ylabel("Error MSE" if GET_MSE else "Error")
    plt.subplots_adjust(left = 0.08, right = 0.99, top = 0.97, bottom = 0.06, 
                        wspace = 0.04, hspace = 0.2)
    if SAVE_FIGS:
        filename = "results/figs/Synth_with_approx"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        if GET_MSE: filename = filename[:-4]+"_MSE.pdf"
        plt.savefig(filename)
    else: plt.show()

# plotEg(1, 'Laplace')
genRandomFunc(20)
# for i in range(len(FUNC_LIST)):
#     plotEg(i, 'Gaussian', eps = 0.1, plotBaseline = True, SAMPLE = 20)
expt('Laplace')
expt('Gaussian')