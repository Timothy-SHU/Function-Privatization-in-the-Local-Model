import sys
from PrivPwcApprox import *
from AdaptApprox import *

EPS = 0.01
BATCH_SIZE = 1000
SVT_THRESHOLD_SCALE = 10
separate = None
parallel = None
interactive = True

if len(sys.argv) > 1:
    EPS = float(sys.argv[1])
    BATCH_SIZE = int(sys.argv[2])
    SVT_THRESHOLD_SCALE = float(sys.argv[3])
    separate = sys.argv[4]
    parallel = sys.argv[5]
    interactive = False
    # interactive = True

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory.pkl")
if interactive:
    print("="*50)
    print(f"Datasets loaded in {time.time()-timer:.5f} sec.")
    print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
    print("="*50)

while separate not in ["Y", "y", "N", "n"]:
    separate = input("Approximate & privatize each coordinate separately? (Y/n)\t")
while parallel not in ["Y", "y", "N", "n"]:
    parallel = input("Enable multiprocessing? (Y/n)\t")
separate = separate in ["Y", "y"]
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
        if separate:
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
        else:
            func = time_series_func_2D(t, x, y)
            solver = adaptive_approx(func = func, interval = (t[0], t[-1]), 
                                     basis = 'Linear-2D', degree = 1, eps = EPS, 
                                     func_2D = (time_series_func(t, x), time_series_func(t, y)), 
                                     SVT_threshold_scale = SVT_THRESHOLD_SCALE, parallel = parallel)
            err_ls = solver.eval('Approx')
            err_priv = solver.eval('Priv')
            priv_loss = solver.evalPrivLoss()
        if interactive:
            approx = solver.createApprox()(t)
            approx_x = approx[:, 0]
            approx_y = approx[:, 1]
            priv = solver.createPriv()(t)
            priv_x = priv[:, 0]
            priv_y = priv[:, 1]
            print(f"Dataset #{i+1} batch #{j+1} done. Executed in {time.time()-iter_timer:.5f} sec.")
            # plt.subplot(2, 1, 1); plt.plot(t, x); plt.plot(t, approx_x); plt.plot(t, priv_x)
            # plt.subplot(2, 1, 2); plt.plot(t, y); plt.plot(t, approx_y); plt.plot(t, priv_y); plt.show()
            plt.plot(x+min_x, y+min_y, color = 'black')
            plt.plot(approx_x+min_x, approx_y+min_y, color = 'red')
            plt.plot(priv_x+min_x, priv_y+min_y, color = 'blue')
            print(f"\t||f-f_approx|| = {err_ls:.5f};", end = " ")
            print(f"||f_approx-f_priv|| = {priv_loss:.5f};", end = " ")
            print(f"||f-f_priv|| = {err_priv:.5f}.")
            break
        iter_timer = time.time()
    if interactive:
        break
if interactive:
    print(f"Total time elapsed: {time.time()-timer:.5f} sec.")
    plt.show()