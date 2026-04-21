import os, sys, time, shutil
import pandas as pd
from PrivPwcApprox import *
from AdaptApprox import *
from tqdm import tqdm

METHOD = 'Laplace'
UNIT_TIME_SCALE = 43200
SVT_THRESHOLD_SCALE = 1
repeat = 30
SAVE_FIGS = True
EPS_LIST = [0.1, 0.01, 0.001]

plt.rc('axes', titlesize = 11)
plt.rc('axes', labelsize = 11)
plt.rc('xtick', labelsize = 9)
plt.rc('ytick', labelsize = 10)
plt.rc('legend', fontsize = 11)

def genNoise(method, scale, dim = 1):
    if method == 'Laplace':
        if dim == 1:
            noise = laplace.rvs(scale = scale)
        else:
            noise = np.random.randn(dim)
            noise = noise/np.linalg.norm(noise)
            noise *= gamma.rvs(a = dim, scale = scale)
    elif method == 'Gaussian':
        if dim == 1: noise = norm.rvs(scale = scale)
        else: noise = scale*np.random.randn(dim)
    return noise

df = pd.read_pickle("cabspottingdata/trajectory_selected.pkl")

timer = time.time()
pbar = tqdm(total = 50)
count = 0
for i in range(len(df)):
# for i in tqdm(range(len(df))):
    for j in range(len(df['t'][i])):
        _, axs = plt.subplots(1, len(EPS_LIST), figsize = (4*len(EPS_LIST), 4), sharex = True, sharey = True)
        for ii in range(len(EPS_LIST)):
            EPS = EPS_LIST[ii]
            t = df['t'][i][j]
            t = np.array([(cur-t[0]).total_seconds() for cur in t])
            eps = EPS/UNIT_TIME_SCALE*(t[-1]-t[0])
            x = np.array(df['x'][i][j])
            y = np.array(df['y'][i][j])
            min_x = np.min(x); x -= min_x
            min_y = np.min(y); y -= min_y
            l = 350; r = l+100
            t = t[l:r]; x = x[l:r]; y = y[l:r]
            func = time_series_func_2D(t, x, y)
            time_series = (t, np.column_stack((x, y)))

            plt.subplot(1, len(EPS_LIST), ii+1)
            plt.plot(x+min_x, y+min_y, color = 'black', label = "True")

            solver, B = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                        basis = 'Linear-2D', degree = 1, 
                                        eps = eps, beta = 0.1, method = METHOD, 
                                        SVT_threshold_scale = SVT_THRESHOLD_SCALE, 
                                        time_series = time_series, parallel = False)
            solver.privatize(B, METHOD)
            solver.smooth()
            smooth = solver.createPriv(); smooth_res = smooth(t)
            smooth_t_x = smooth_res[:, 0]; smooth_t_y = smooth_res[:, 1]
            plt.plot(smooth_t_x+min_x, smooth_t_y+min_y, color = 'tab:blue', 
                     alpha = 0.9, label = "PrivFuncSeg")
            
            solver, B = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                        basis = 'Linear-2D', degree = 1, 
                                        eps = eps, beta = 0.1, method = METHOD, 
                                        SVT_threshold_scale = SVT_THRESHOLD_SCALE, 
                                        time_series = time_series, parallel = False,
                                        enable_reduce_seg = False)
            solver.privatize(B, METHOD)
            solver.smooth()
            smooth = solver.createPriv(); smooth_res = smooth(t)
            smooth_t_x = smooth_res[:, 0]; smooth_t_y = smooth_res[:, 1]
            plt.plot(smooth_t_x+min_x, smooth_t_y+min_y, color = 'tab:orange', 
                     alpha = 0.9, label = "PrivFuncSeg\n(w/o ReduceSeg)")
            plt.title(f"eps = {EPS}")

            plt.legend()

        plt.subplots_adjust(left = 0.07, right = 0.975, top = 0.925, bottom = 0.11, 
                            wspace = 0.07, hspace = 0.2)
        # plt.show()
        if SAVE_FIGS:
            plt.savefig(f"results/figs/ReduceSeg/Taxi_{i}.pdf")
        plt.close()

        pbar.update(1)
        count += 1
        if count == 50: exit(0)