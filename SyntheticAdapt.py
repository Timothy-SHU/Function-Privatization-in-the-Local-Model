import os, sys, random, tqdm
from PrivPwcApprox import *
from AdaptBasis import *

repeat = 50
SAVE_FIGS = True
GET_MSE = False

plt.rc('axes', titlesize = 13)
plt.rc('axes', labelsize = 13)
plt.rc('xtick', labelsize = 11)
plt.rc('ytick', labelsize = 11)
plt.rc('legend', fontsize = 13)

def getMSE(arr):
    return (arr**2).mean()
    # avg = arr.mean()
    # return ((arr-avg)**2).mean()

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
SAMPLE_LIST = [10, 20, 50, 100]
WINDOW_SCALE = 0.1

EPS_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
RHO_LIST = [1e-7, 4e-7, 2.5e-6, 1e-5, 4e-5, 2.5e-4, 0.001]

# EPS_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
# RHO_LIST = [1e-7, 4e-7, 2.5e-6, 1e-5, 4e-5, 2.5e-4, 0.001, 0.004, 0.025, 0.1]

def genRandomFunc(n, SEED = 42):
    global FUNC_LIST
    FUNC_LIST = []; random.seed(SEED)
    for i in range(n):
        params = []; m = random.randint(1, 5)
        for j in range(m):
            params.append((random.randint(50, 500), random.randint(0, 100), random.randint(2, 16)))
        # print(params)
        FUNC_LIST.append(genGaus(params))

def plotEg(idx, method, eps = 0.1, plotBaseline = False, SAMPLE = 10):
    func = FUNC_LIST[idx]
    if method == 'Gaussian': eps = eps*eps/10
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

    plt.figure(figsize = (8, 6))
    solver, deg = adaptive_basis(func = func, interval = (0, 100), 
                                 eps = eps, method = method)
    print(f"||f|| = {np.sqrt(solver.funcSqrInt):.5f};")
    print(f"||f-f_approx|| = {solver.eval('Approx'):.5f};")
    print(f"||f_approx-f_priv|| = {solver.evalPrivLoss():.5f};")
    print(f"||f-f_priv|| = {solver.eval('Priv'):.5f};")
    integrand = lambda x: (func(x)-np.interp(x, sample, val_priv))**2
    print(f"||f-f_bl|| = {np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0]):.5f};")
    integrand = lambda x: (func(x)-np.interp(x, sample, val_smooth))**2
    print(f"||f-f_bl_sm|| = {np.sqrt(quad(integrand, 0, 100, limit = INTLIM)[0]):.5f}.\n")

    x_dense = np.linspace(0, 100, INTLIM)
    plt.plot(x_dense, func(x_dense), color = 'black', label = "Function")
    if plotBaseline:
        plt.plot(sample, val_priv, color = 'tab:green', alpha = 0.9, label = "Baseline")
        plt.plot(sample, val_smooth, color = 'tab:purple', alpha = 0.9, label = "Baseline (smoothed)")
    plt.plot(x_dense, solver.createApprox()(x_dense), color = 'tab:brown', alpha = 0.9, label = "LS Approximation")
    plt.plot(x_dense, solver.createPriv()(x_dense), color = 'tab:blue', alpha = 0.9, label = "PrivFuncSelect")
    plt.title(f"Monomial Basis with Degrees: {deg}"); plt.legend(loc = 'upper left')
    plt.show()

def expt(method):
    results = np.zeros((8, len(EPS_LIST)))
    for idx, EPS in enumerate(EPS_LIST):
        eps = EPS
        if method == 'Gaussian': eps = eps*eps/10
        filename = f"results/Synth/SynthAdapt_{EPS}_{method}.txt"
        if not os.path.isfile(filename):
        # if True:
            print(f"Running {'GP' if method == 'Laplace' else 'CGP'} with eps = {EPS}...")
            pbar = tqdm.tqdm(total = len(FUNC_LIST)*repeat*(1+len(SAMPLE_LIST)))
            for func in FUNC_LIST:
                funcL2 = None
                err_ls = []; err_priv = []; err_smooth = []
                for k in range(repeat):
                    solver, _ = adaptive_basis(func = func, interval = (0, 100), 
                                               eps = eps, method = method)
                    if funcL2 == None: funcL2 = np.sqrt(solver.funcSqrInt)
                    err_ls.append(solver.eval('Approx')/funcL2)
                    err_priv.append(solver.eval('Priv')/funcL2)
                    pbar.update(1)
                err_ls = np.array(err_ls)
                err_priv = np.array(err_priv)
                results[0, idx] += err_ls.mean()/len(FUNC_LIST)
                results[1, idx] += err_priv.mean()/len(FUNC_LIST)
                results[4, idx] += getMSE(err_ls)/len(FUNC_LIST)
                results[5, idx] += getMSE(err_priv)/len(FUNC_LIST)

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
                                val_priv[i] += genNoise(method, SAMPLE/eps)
                            elif method == 'Gaussian':
                                val_priv[i] += genNoise(method, np.sqrt(SAMPLE/(2*eps)))
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
                    if min_err_priv == None or err_priv.mean() < min_err_priv:
                        min_err_priv = err_priv.mean()
                        min_err_priv_MSE = getMSE(err_priv)
                    if min_err_smooth == None or err_smooth.mean() < min_err_smooth:
                        min_err_smooth = err_smooth.mean()
                        min_err_smooth_MSE = getMSE(err_smooth)
                results[2, idx] += min_err_priv/len(FUNC_LIST)
                results[3, idx] += min_err_smooth/len(FUNC_LIST)
                results[6, idx] += min_err_priv_MSE/len(FUNC_LIST)
                results[7, idx] += min_err_smooth_MSE/len(FUNC_LIST)
            os.makedirs("results/Synth/", exist_ok = True)
            res_file = open(filename, 'w')
            for k in range(8):
                res_file.write(f"{results[k, idx]} ")
            res_file.write("\n")
            res_file.close()
            pbar.close()
        else:
            print(f"Loading results for {'GP' if method == 'Laplace' else 'CGP'} with eps = {EPS}...", end = '\t')
            file = open(filename, 'r')
            buffer = [float(x) for x in file.read().strip().split()]
            for k in range(8):
                results[k, idx] = buffer[0]
                buffer = buffer[1:]
            print("Complete.")
    
    if GET_MSE: results = results[4:, :]
    else: results = results[:4, :]

    names = []; colors = []; markers = []
    names.append("LS Approximation"); colors.append('tab:brown'); markers.append('P')
    names.append("PrivFuncSelect"); colors.append('tab:blue'); markers.append('^')
    names.append("Baseline"); colors.append('tab:green'); markers.append('s')
    names.append("Baseline (smoothed)"); colors.append('tab:purple'); markers.append('d')

    budgets = EPS_LIST if method == 'Laplace' else RHO_LIST
    plt.figure(figsize = (5, 4))
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    for idx in range(1, len(names)):
        plt.plot(budgets, results[idx, :].tolist(), color = colors[idx], 
                 alpha = 0.9, marker = markers[idx], label = names[idx])
    plt.legend(loc = 'upper right'); plt.xscale('log'); plt.yscale('log')
    plt.xticks(budgets, budgets, minor = False)
    if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
    elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
    # plt.ylabel("Error MSE" if GET_MSE else "Error")
    if method == 'Gaussian' and GET_MSE: plt.margins(y = 0.06)
    plt.subplots_adjust(left = 0.075, right = 0.99, top = 0.99, bottom = 0.12)
    if GET_MSE: plt.subplots_adjust(left = 0.09)
    if SAVE_FIGS:
        filename = "results/figs/SynthAdapt"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        if GET_MSE: filename = filename[:-4]+"_MSE.pdf"
        plt.savefig(filename)

    plt.figure(figsize = (6, 4))
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    for idx in range(len(names)):
        plt.plot(budgets, results[idx, :].tolist(), color = colors[idx], 
                 alpha = 0.9, marker = markers[idx], label = names[idx])
    plt.legend(loc = 'upper right'); plt.xscale('log'); plt.yscale('log')
    plt.xticks(budgets, budgets, minor = False)
    if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
    elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
    plt.ylabel("Error MSE" if GET_MSE else "Error")
    plt.subplots_adjust(left = 0.1, right = 0.99, top = 0.99, bottom = 0.12)
    if method == 'Gaussian': plt.subplots_adjust(left = 0.1)
    if GET_MSE: plt.subplots_adjust(left = 0.115)
    if SAVE_FIGS:
        filename = "results/figs/SynthAdapt_with_approx"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        if GET_MSE: filename = filename[:-4]+"_MSE.pdf"
        plt.savefig(filename)
    else: plt.show()

genRandomFunc(20)
# for i in [1, 3, 5, 7, 9]:
#     plotEg(i, 'Gaussian', eps = 1.0, plotBaseline = True, SAMPLE = 20)
for i in range(2):
    expt('Laplace')
    expt('Gaussian')
    GET_MSE = True