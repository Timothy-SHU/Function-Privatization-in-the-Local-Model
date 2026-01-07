import os, sys
import numpy as np
import matplotlib.pyplot as plt

N = 1000
FREQUENCY = 100
TIME_SCALE = 80
ECG_RECORDS = 100
UNIT_TIME_SCALE = 43200
repeat = 30
SAVE_FIGS = True
GET_MSE = False

plt.rc('axes', titlesize = 11)
plt.rc('axes', labelsize = 11)
plt.rc('xtick', labelsize = 10)
plt.rc('ytick', labelsize = 10)
plt.rc('legend', fontsize = 9)

def getStats(filename, isBaseline, adaptive, smoothed):
    file = open(filename, 'r')
    buffer = [float(x) for x in file.read().strip().split()]
    funcL2 = buffer[2]; buffer = buffer[3:]
    if (not isBaseline) and (not adaptive):
        err_ls = buffer[0] / funcL2
        buffer = buffer[2:]
    res = []
    for i in range(repeat):
        rec = []
        if adaptive:
            rec += [buffer[0]/funcL2]
            buffer = buffer[2:]
        if isBaseline:
            rec += [buffer[0]/funcL2, buffer[2]/funcL2]
            buffer = buffer[4:]
        elif smoothed:
            rec += [buffer[1]/funcL2, buffer[4]/funcL2]
            buffer = buffer[6:]
        else:
            rec += [buffer[1]/funcL2]
            buffer = buffer[3:]
        res.append(rec)
    res = np.array(res)
    avg = res.mean(axis = 0)
    if GET_MSE:
        avg = ((res-avg)**2).mean(axis = 0)
    if isBaseline or adaptive: return avg.tolist()
    return [err_ls] + avg.tolist()

def getTaxiRes(METHOD, EPS):
    dir = f"results/TaxiTrajectory/taxi_{EPS}/"
    num_curve = 0; stats_sum = np.zeros(3)
    for subdir in sorted(os.scandir(dir), key = lambda e: e.name):
        if not subdir.is_dir(): continue
        folder = os.path.basename(subdir.path)+"/"
        for file in os.listdir(dir+folder):
            if METHOD in file:
                num_curve += 1
                stats_sum += getStats(dir+folder+file, False, True, True)
    return stats_sum/num_curve

def getTaxiBLRes(METHOD, EPS, SAMPLE_RATE, WINDOW_SCALE):
    dir = f"results/TaxiTrajectory/taxi_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    num_curve = 0; stats_sum = np.zeros(2)
    for subdir in sorted(os.scandir(dir), key = lambda e: e.name):
        if not subdir.is_dir(): continue
        folder = os.path.basename(subdir.path)+"/"
        for file in os.listdir(dir+folder):
            if METHOD in file:
                num_curve += 1
                stats_sum += getStats(dir+folder+file, True, False, True)
    return stats_sum/num_curve

def getECGRes(METHOD, EPS, BATCH_SIZE):
    dir = f"results/ECG/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}/"
    if BATCH_SIZE == -1: dir = f"results/ECG/ECG_{EPS}_unbounded/"
    num_rec = 0; stats_sum = np.zeros(3)
    for i in range(22):
        folder = "{:05d}/".format(i*1000)
        for j in range(1000):
            file = f"{i*1000+j:05d}_lr_{METHOD}.txt"
            if os.path.exists(dir+folder+file):
                num_rec += 1
                stats_sum += getStats(dir+folder+file, False, False, True)
            if num_rec == ECG_RECORDS: break
        if num_rec == ECG_RECORDS: break
    return stats_sum/num_rec

def getECGBLRes(METHOD, EPS, SAMPLE_RATE, WINDOW_SCALE):
    dir = f"results/ECG/ECG_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    num_rec = 0; stats_sum = np.zeros(2)
    for i in range(22):
        folder = "{:05d}/".format(i*1000)
        for j in range(1000):
            file = f"{i*1000+j:05d}_lr_{METHOD}.txt"
            if os.path.exists(dir+folder+file):
                num_rec += 1
                stats_sum += getStats(dir+folder+file, True, False, True)
            if num_rec == ECG_RECORDS: break
        if num_rec == ECG_RECORDS: break
    return stats_sum/num_rec

def plotRes(isTaxi, method, unbounded = False):
    SAMPLE_RATE_LIST = [0.1, 0.2]
    WINDOW_SCALE_LIST = [0.05, 0.1]
    if not isTaxi: SAMPLE_RATE_LIST.append(0.8)

    names = []; results = []; colors = []; markers = []
    names.append("LS Approximation"); results.append([])
    colors.append('tab:brown'); markers.append('P')
    names.append("Privatization"); results.append([])
    colors.append('tab:blue'); markers.append('^')
    names.append("Privatization (continuous)"); results.append([])
    colors.append('tab:orange'); markers.append('o')
    names.append("Baseline"); results.append([])
    colors.append('tab:purple'); markers.append('s')
    names.append("Baseline (smoothed)"); results.append([])
    colors.append('tab:green'); markers.append('d')

    if isTaxi:
        names[1] = "PrivFuncSeg"
        names[2] = "PrivFuncSeg (continuous)"

    if isTaxi:
        EPS_LIST = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        RHO_LIST = [1.25e-8, 5e-8, 3.125e-7, 1.25e-6, 5e-6, 3.125e-5, 1.25e-4]
    else:
        EPS_LIST = [0.1, 0.2, 0.5, 1.0, 2.0]
        RHO_LIST = [8e-5, 3.2e-4, 2e-3, 8e-3, 3.2e-2]

    for EPS in EPS_LIST:
        if isTaxi: res = getTaxiRes(method, EPS)
        elif unbounded: res = getECGRes(method, EPS, -1)
        else: res = getECGRes(method, EPS, 20)
        for idx in range(3):
            results[idx].append(res[idx])
        results[3].append(None); results[4].append(None)
        for SAMPLE_RATE in SAMPLE_RATE_LIST:
            for WINDOW_SCALE in WINDOW_SCALE_LIST:
                if isTaxi: res = getTaxiBLRes(method, EPS, SAMPLE_RATE, WINDOW_SCALE)
                else: res = getECGBLRes(method, EPS, SAMPLE_RATE, WINDOW_SCALE)
                if results[3][-1] == None: results[3][-1] = res[0]
                else: results[3][-1] = min(results[3][-1], res[0])
                if results[4][-1] == None: results[4][-1] = res[1]
                else: results[4][-1] = min(results[4][-1], res[1])

    budgets = EPS_LIST if method == 'Laplace' else RHO_LIST
    plt.figure(figsize = (6, 4))
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    for idx in range(len(names)):
        if not isTaxi and idx == 0: continue
        if unbounded and idx == 2: continue
        plt.plot(budgets, results[idx], color = colors[idx], 
                 alpha = 0.9, marker = markers[idx], label = names[idx])
    plt.legend(loc = 'upper right'); plt.xscale('log'); plt.yscale('log')
    plt.xticks(budgets, budgets, minor = False)
    if method == 'Laplace': plt.xlabel("Privacy Budget "+r"$\varepsilon$")
    elif method == 'Gaussian': plt.xlabel("Privacy Budget "+r"$\rho$")
    if GET_MSE: plt.ylabel("Error MSE")
    else: plt.ylabel("Error")
    plt.subplots_adjust(left = 0.11, right = 0.98, top = 0.99, bottom = 0.13)
    if SAVE_FIGS:
        if isTaxi: filename = "results/figs/Taxi"
        elif not unbounded: filename = "results/figs/ECG"
        else: filename = "results/figs/ECG_unbounded"
        filename += "_GP.pdf" if method == 'Laplace' else "_CGP.pdf"
        if GET_MSE: filename = filename[:-4]+"_MSE.pdf"
        plt.savefig(filename)
    else: plt.show()

for i in range(2):
    plotRes(True, 'Laplace')
    plotRes(True, 'Gaussian')
    plotRes(False, 'Laplace')
    plotRes(False, 'Gaussian')
    plotRes(False, 'Laplace', True)
    plotRes(False, 'Gaussian', True)
    GET_MSE = True