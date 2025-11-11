from PrivPwcApprox import *
from AdaptApprox import *
from ast import literal_eval

UNIT = 1000
EARTH_RADIUS = 6371*UNIT
SPEED_LIMIT = 180*UNIT/60/60

BATCH_SIZE = 100
SVT_THRESHOLD_SCALE = 200

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory.pkl")
print("="*50)
print(f"Datasets loaded in {time.time()-timer:.5f} sec.")
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)

mode = input("Approximate & privatize each coordinate separately? (Y/n)\t")
while mode not in ["Y", "y", "N", "n"]:
    mode = input("Approximate & privatize each coordinate separately? (Y/n)\t")

timer = time.time()
iter_timer = time.time()
# for i in range(len(df)):
for i in range(1):
    # for j in range(len(df['t'][i])//BATCH_SIZE):
    for j in range(10):
        t = df['t'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        x = df['x'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y = df['y'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        min_x = np.min(x); x -= min_x
        min_y = np.min(y); y -= min_y
        if mode == "Y" or mode == "y":
            func_x = time_series_func(t, x)
            func_y = time_series_func(t, y)
            solver_x = adaptive_poly_approx(func = func_x, interval = (t[0], t[-1]), degree = 1, 
                                            eps = 0.5, SVT_threshold_scale = SVT_THRESHOLD_SCALE)
            solver_y = adaptive_poly_approx(func = func_y, interval = (t[0], t[-1]), degree = 1, 
                                            eps = 0.5, SVT_threshold_scale = SVT_THRESHOLD_SCALE)
            approx_x = solver_x.createApprox()
            priv_x = solver_x.createPriv()
            approx_y = solver_y.createApprox()
            priv_y = solver_y.createPriv()
            # logging.info(f"Approx x: {approx_x(t)}\nApprox y: {approx_y(t)}\n\n")
            # print(l2_dist(func_x, approx_x, t[0], t[-1]), l2_dist(func_x, priv_x, t[0], t[-1]))
            # print(l2_dist(func_y, approx_y, t[0], t[-1]), l2_dist(func_y, priv_y, t[0], t[-1]))
            plt.plot(x+min_x, y+min_y, color = 'black')
            plt.plot(approx_x(t)+min_x, approx_y(t)+min_y, color = 'red')
            plt.plot(priv_x(t)+min_x, priv_y(t)+min_y, color = 'blue')
        else:
            func = time_series_func_2D(t, x, y)
            solver = adaptive_poly_approx(func = func, interval = (t[0], t[-1]), 
                                          basis = 'Linear-2D', degree = 1, eps = 1, 
                                          SVT_threshold_scale = SVT_THRESHOLD_SCALE)
            approx = solver.createApprox()(t)
            approx_x = approx[:, 0]
            approx_y = approx[:, 1]
            priv = solver.createPriv()(t)
            priv_x = priv[:, 0]
            priv_y = priv[:, 1]
            plt.plot(x+min_x, y+min_y, color = 'black')
            plt.plot(approx_x+min_x, approx_y+min_y, color = 'red')
            plt.plot(priv_x+min_x, priv_y+min_y, color = 'blue')
        print(f"Dataset #{i+1} batch #{j+1} done. Executed in {time.time()-iter_timer:.5f} sec.")
        iter_timer = time.time()
print(f"Total time elapsed: {time.time()-timer:.5f} sec.")
plt.show()