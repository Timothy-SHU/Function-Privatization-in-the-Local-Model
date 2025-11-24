import sys
import shutil
from PrivPwcApprox_quad import *
from AdaptApprox_quad import *

EPS = 0.01
BATCH_SIZE = 1000
SVT_THRESHOLD_SCALE = 10
repeat = 20
parallel = None
interactive = True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EPS = float(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
        SVT_THRESHOLD_SCALE = int(sys.argv[3])
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
    iter_timer = time.time()
    for i in range(len(df)):
        for j in range(len(df['t'][i])//BATCH_SIZE):
            t = df['t'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            t = [(cur-t[0]).total_seconds() for cur in t]
            x = df['x'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            y = df['y'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
            min_x = np.min(x); x -= min_x
            min_y = np.min(y); y -= min_y
            func = time_series_func_2D(t, x, y)
            solver = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                     basis = 'Linear-2D', degree = 1, eps = EPS, 
                                     func_2D = (time_series_func(t, x), time_series_func(t, y)), 
                                     SVT_threshold_scale = SVT_THRESHOLD_SCALE, parallel = parallel)
            err_ls = solver.eval('Approx')
            err_priv = solver.eval('Priv')
            priv_loss = solver.evalPrivLoss()
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
                    plt.plot(approx_x+min_x, approx_y+min_y, color = 'red')
                    plt.plot(priv_x+min_x, priv_y+min_y, color = 'blue')
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
                    plt.plot(approx_x+min_x, approx_y+min_y, color = 'red')
                    plt.plot(smooth_x+min_x, smooth_y+min_y, color = 'blue')
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
                plt.plot(t, approx_t_x+min_x, color = 'red')
                plt.plot(t, priv_t_x+min_x, color = 'blue')
                plt.plot(t, smooth_t_x+min_x, color = 'green')
                plt.subplot(2, 1, 2)
                plt.plot(t, y+min_y, color = 'black')
                plt.plot(t, approx_t_y+min_y, color = 'red')
                plt.plot(t, priv_t_y+min_y, color = 'blue')
                plt.plot(t, smooth_t_y+min_y, color = 'green')
                plt.show()
                break
            else:
                filename = df['filename'][i].removeprefix("new_").removesuffix(".txt")
                dir = f"results/taxi_{EPS}_{BATCH_SIZE}_{SVT_THRESHOLD_SCALE}/{filename}/"
                os.makedirs(dir, exist_ok = True)
                res_file = open(dir+f"{filename}-{j+1}.txt", 'w')
                res_file.write(f"{np.sqrt(solver.funcSqrInt)}\n")
                res_file.write(f"{err_ls}\n")
                res_file.write(f"{priv_loss} {err_priv}\n")
                for k in range(repeat-1):
                    solver.privatize(EPS)
                    err_priv = solver.eval('Priv')
                    priv_loss = solver.evalPrivLoss()
                    res_file.write(f"{priv_loss} {err_priv}\n")
                res_file.close()
                print(f"Dataset #{i+1} batch #{j+1} done. Executed in {time.time()-iter_timer:.2f} sec.", flush = True)
            iter_timer = time.time()

            """ Privatize x and y separately
            func_x = time_series_func(t, x)
            func_y = time_series_func(t, y)
            solver_x = adaptive_approx(func = func_x, interval = (t[0], t[-1]), degree = 1, eps = EPS/2, 
                                       SVT_threshold_scale = SVT_THRESHOLD_SCALE, parallel = parallel)
            solver_y = adaptive_approx(func = func_y, interval = (t[0], t[-1]), degree = 1, eps = EPS/2, 
                                       SVT_threshold_scale = SVT_THRESHOLD_SCALE, parallel = parallel)
            approx_x = solver_x.createApprox()
            priv_x = solver_x.createPriv()
            approx_y = solver_y.createApprox()
            priv_y = solver_y.createPriv()
            err_ls = np.sqrt(solver_x.eval('Approx')**2+solver_y.eval('Approx')**2)
            err_priv = np.sqrt(solver_x.eval('Priv')**2+solver_y.eval('Priv')**2)
            priv_loss = np.sqrt(solver_x.evalPrivLoss()**2+solver_y.evalPrivLoss()**2)
            plt.plot(x+min_x, y+min_y, color = 'black')
            plt.plot(approx_x(t)+min_x, approx_y(t)+min_y, color = 'red')
            plt.plot(priv_x(t)+min_x, priv_y(t)+min_y, color = 'blue')
            """
        if interactive:
            break

        # break   # sample: run only the first dataset

    if not interactive:
        dir = f"results/taxi_{EPS}_{BATCH_SIZE}_{SVT_THRESHOLD_SCALE}/"
        shutil.copyfile("info.log", dir+"info.log")
        print(f"Total time elapsed: {time.time()-timer:.2f} sec.")