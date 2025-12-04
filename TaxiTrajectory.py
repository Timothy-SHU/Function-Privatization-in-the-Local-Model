import os, sys, time, shutil
import pandas as pd
from PrivPwcApprox import *
from AdaptApprox import *
from tqdm import tqdm

EPS = 0.01
METHOD = 'Laplace'
UNIT_TIME_SCALE = 43200
SVT_THRESHOLD_SCALE = 1.0
repeat = 10
parallel = None
interactive = True

if len(sys.argv) > 1:
    METHOD = str(sys.argv[1])
    EPS = float(sys.argv[2])
    SVT_THRESHOLD_SCALE = float(sys.argv[3])
    parallel = False
    interactive = False
    # interactive = True

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory_selected.pkl")
print("="*80)
print(f"Datasets loaded in {time.time()-timer:.2f} sec.")
print(f"Total # of curves: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*80)

if len(sys.argv) == 1:
    str = input("Privacy budget per 12h (default 0.01):\t")
    if str != "": EPS = float(str)
    # str = input("SVT threshold scale (default 1):\t")
    # if str != "": SVT_THRESHOLD_SCALE = float(str)
    parallel = input("Enable multiprocessing? [y/N]\t")
    parallel = parallel in ["Y", "y"]

if SVT_THRESHOLD_SCALE.is_integer():
    SVT_THRESHOLD_SCALE = int(SVT_THRESHOLD_SCALE)

timer = time.time()
pbar = tqdm(total = np.sum([len(df['t'][i]) for i in range(len(df))]))
for i in range(len(df)):
# for i in tqdm(range(len(df))):
    for j in range(len(df['t'][i])):
        iter_timer = time.time()
        t = df['t'][i][j]
        t = np.array([(cur-t[0]).total_seconds() for cur in t])
        eps = EPS/UNIT_TIME_SCALE*(t[-1]-t[0])
        x = np.array(df['x'][i][j])
        y = np.array(df['y'][i][j])
        min_x = np.min(x); x -= min_x
        min_y = np.min(y); y -= min_y
        func = time_series_func_2D(t, x, y)
        time_series = (t, np.column_stack((x, y)))

        if interactive:
            approx_timer = time.time()
            solver = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                    basis = 'Linear-2D', degree = 1, 
                                    eps = eps, beta = 0.1, method = METHOD, 
                                    SVT_threshold_scale = SVT_THRESHOLD_SCALE, 
                                    time_series = time_series, parallel = parallel)
            approx_time = time.time()-approx_timer
            err_ls = solver.eval('Approx')
            err_priv = solver.eval('Priv')
            priv_loss = solver.evalPrivLoss()
            approx = solver.createApprox(); approx_res = approx(t)
            approx_t_x = approx_res[:, 0]; approx_t_y = approx_res[:, 1]
            priv = solver.createPriv(); priv_res = priv(t)
            priv_t_x = priv_res[:, 0]; priv_t_y = priv_res[:, 1]

            print(f"Time range: {t[-1]-t[0]} sec; eps = {eps}.")
            plt.subplot(1, 2, 1)
            plt.plot(x+min_x, y+min_y, color = 'black')
            for k in range(len(solver.breakpoints)-1):
                l = solver.breakpoints[k]; r = solver.breakpoints[k+1]
                dense_t = np.linspace(l, r, INTLIM_PER_PIECE)[:-1]
                approx_res = approx(dense_t)
                approx_x = approx_res[:, 0]; approx_y = approx_res[:, 1]
                priv_res = priv(dense_t)
                priv_x = priv_res[:, 0]; priv_y = priv_res[:, 1]
                plt.plot(approx_x+min_x, approx_y+min_y, color = 'tab:blue')
                plt.plot(priv_x+min_x, priv_y+min_y, color = 'tab:orange')
            print(f"Privatized in {time.time()-iter_timer:.2f} sec.")

            smooth_timer = time.time()
            solver.smooth()
            err_smooth = solver.eval('Priv')
            smooth_loss = solver.evalPrivLoss()
            smooth = solver.createPriv(); smooth_res = smooth(t)
            smooth_t_x = smooth_res[:, 0]; smooth_t_y = smooth_res[:, 1]

            plt.subplot(1, 2, 2)
            plt.plot(x+min_x, y+min_y, color = 'black')
            for k in range(len(solver.breakpoints)-1):
                l = solver.breakpoints[k]; r = solver.breakpoints[k+1]
                dense_t = np.linspace(l, r, INTLIM_PER_PIECE)[:-1]
                approx_res = approx(dense_t)
                approx_x = approx_res[:, 0]; approx_y = approx_res[:, 1]
                smooth_res = smooth(dense_t)
                smooth_x = smooth_res[:, 0]; smooth_y = smooth_res[:, 1]
                # plt.plot(approx_x+min_x, approx_y+min_y, color = 'tab:blue')
                plt.plot(smooth_x+min_x, smooth_y+min_y, color = 'tab:green')
            print(f"Smoothed in {time.time()-smooth_timer:.2f} sec.")
            print(f"Total time elapsed: {time.time()-iter_timer:.2f} sec.")
            print(f"||f-f_approx|| = {err_ls:.5f};")
            print(f"||f_approx-f_priv|| = {priv_loss:.5f};", end = " ")
            print(f"||f-f_priv|| = {err_priv:.5f};")
            print(f"||f_approx-f_smooth|| = {smooth_loss:.5f};", end = " ")
            print(f"||f-f_smooth|| = {err_smooth:.5f};")
            print("="*80)
            plt.show()

            plt.subplot(2, 1, 1)
            plt.plot(t, x+min_x, color = 'black')
            plt.plot(t, approx_t_x+min_x, color = 'tab:orange')
            plt.plot(t, priv_t_x+min_x, color = 'tab:blue')
            plt.plot(t, smooth_t_x+min_x, color = 'tab:green')
            plt.subplot(2, 1, 2)
            plt.plot(t, y+min_y, color = 'black')
            plt.plot(t, approx_t_y+min_y, color = 'tab:red')
            plt.plot(t, priv_t_y+min_y, color = 'tab:blue')
            plt.plot(t, smooth_t_y+min_y, color = 'tab:green')
            plt.show()
            break

        else:
            filename = df['filename'][i].removeprefix("new_").removesuffix(".txt")
            dir = f"results/TaxiTrajectory/taxi_{EPS}/{filename}/"
            os.makedirs(dir, exist_ok = True)
            res_file = open(dir+f"{filename}-{j+1}_{METHOD}.txt", 'w')
            res_file.write(f"{t[-1]-t[0]} {eps}\n")
            for k in range(repeat):
                approx_timer = time.time()
                solver = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                         basis = 'Linear-2D', degree = 1, 
                                         eps = eps, beta = 0.1, method = METHOD, 
                                         SVT_threshold_scale = SVT_THRESHOLD_SCALE, 
                                         time_series = time_series, parallel = parallel)
                approx_time = time.time()-approx_timer
                err_ls = solver.eval('Approx')
                err_priv = solver.eval('Priv')
                priv_loss = solver.evalPrivLoss()
                priv_timer = time.time()
                solver.privatize(eps, METHOD)
                priv_time = time.time()-priv_timer
                err_priv = solver.eval('Priv')
                priv_loss = solver.evalPrivLoss()
                if k == 0:
                    res_file.write(f"{np.sqrt(solver.funcSqrInt)}\n\n")
                res_file.write(f"{err_ls} {approx_time}\n")
                res_file.write(f"{priv_loss} {err_priv} {priv_time}\n")
                smooth_timer = time.time()
                solver.smooth()
                smooth_time = time.time()-smooth_timer
                err_smooth = solver.eval('Priv')
                smooth_loss = solver.evalPrivLoss()
                res_file.write(f"{smooth_loss} {err_smooth} {smooth_time}\n\n")
            res_file.close()
            print(f"Dataset #{i+1} curve #{j+1} done. Executed in {time.time()-iter_timer:.2f} sec.", flush = True)
            pbar.update(1)

    if interactive: break
    if i == 19: break   # sample: run only the first twenty datasets

if not interactive:
    dir = f"results/TaxiTrajectory/taxi_{EPS}/"
    os.makedirs(dir, exist_ok = True)
    shutil.copyfile("info.log", dir+f"info_{METHOD}.log")
    print(f"Total time elapsed: {time.time()-timer:.2f} sec.")