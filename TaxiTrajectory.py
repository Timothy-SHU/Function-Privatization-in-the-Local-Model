import os, sys, time, shutil
import pandas as pd
from PrivPwcApprox import *
from AdaptApprox import *

EPS = 0.01
BATCH_SIZE = 1000
SVT_THRESHOLD_SCALE = 10
repeat = 20
parallel = None
interactive = True

if len(sys.argv) > 1:
    EPS = float(sys.argv[1])
    BATCH_SIZE = int(sys.argv[2])
    SVT_THRESHOLD_SCALE = float(sys.argv[3])
    if SVT_THRESHOLD_SCALE.is_integer():
        SVT_THRESHOLD_SCALE = int(SVT_THRESHOLD_SCALE)
    parallel = sys.argv[4]
    interactive = False
    # interactive = True

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory.pkl")
print("="*50)
print(f"Datasets loaded in {time.time()-timer:.2f} sec.")
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)

while parallel not in ["Y", "y", "N", "n"]:
    parallel = input("Enable multiprocessing? (Y/n)\t")
parallel = parallel in ["Y", "y"]

timer = time.time()
for i in range(len(df)):
    for j in range((len(df['t'][i])-1)//BATCH_SIZE+1):
        iter_timer = time.time()
        t = df['t'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
        t = [(cur-t[0]).total_seconds() for cur in t]
        x = df['x'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
        y = df['y'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
        min_x = np.min(x); x -= min_x
        min_y = np.min(y); y -= min_y
        func = time_series_func_2D(t, x, y)
        time_series = (t, np.column_stack((x, y)))
        solver = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                 basis = 'Linear-2D', degree = 1, eps = EPS, 
                                 SVT_threshold_scale = SVT_THRESHOLD_SCALE, 
                                 time_series = time_series, parallel = parallel)
        err_ls = solver.eval('Approx')
        err_priv = solver.eval('Priv')
        priv_loss = solver.evalPrivLoss()
        approx_time = time.time()-iter_timer

        if interactive:
            approx = solver.createApprox(); approx_res = approx(t)
            approx_t_x = approx_res[:, 0]; approx_t_y = approx_res[:, 1]
            priv = solver.createPriv(); priv_res = priv(t)
            priv_t_x = priv_res[:, 0]; priv_t_y = priv_res[:, 1]

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
            dir = f"results/taxi_{EPS}_{BATCH_SIZE}_{SVT_THRESHOLD_SCALE}/{filename}/"
            os.makedirs(dir, exist_ok = True)
            res_file = open(dir+f"{filename}-{j+1}.txt", 'w')
            res_file.write(f"{np.sqrt(solver.funcSqrInt)}\n")
            res_file.write(f"{err_ls} {approx_time}\n\n")
            for k in range(repeat):
                priv_timer = time.time()
                solver.privatize(EPS)
                err_priv = solver.eval('Priv')
                priv_loss = solver.evalPrivLoss()
                priv_time = time.time()-priv_timer
                res_file.write(f"{priv_loss} {err_priv} {priv_time}\n")
                smooth_timer = time.time()
                solver.smooth()
                err_smooth = solver.eval('Priv')
                smooth_loss = solver.evalPrivLoss()
                smooth_time = time.time()-smooth_timer
                res_file.write(f"{smooth_loss} {err_smooth} {smooth_time}\n\n")
            res_file.close()
            print(f"Dataset #{i+1} batch #{j+1} done. Executed in {time.time()-iter_timer:.2f} sec.", flush = True)
    if interactive:
        break
    # break   # sample: run only the first dataset

if not interactive:
    dir = f"results/taxi_{EPS}_{BATCH_SIZE}_{SVT_THRESHOLD_SCALE}/"
    shutil.copyfile("info.log", dir+"info.log")
    print(f"Total time elapsed: {time.time()-timer:.2f} sec.")