from PrivPwcApprox import *

UNIT = 1000
EARTH_RADIUS = 6371*UNIT
SPEED_LIMIT = 180*UNIT/60/60

BATCH_SIZE = 100
SVT_THRESHOLD_SCALE = 200

df = pd.read_pickle("cabspottingdata/trajectory.pkl")
print("="*50)
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)

# for i in range(len(df)):
for i in range(10):
    # for j in range(len(df['t'][i])//BATCH_SIZE):
    for j in range(10):
        t = df['t'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        x = df['x'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y = df['y'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        min_x = np.min(x); x -= min_x
        min_y = np.min(y); y -= min_y
        func_x = time_series_func(t, x)
        integrand_x = lambda t: func_x(t)**2
        integral_x, _ = quad(integrand_x, t[0], t[-1], limit = INTLIM)
        func_y = time_series_func(t, y)
        integrand_y = lambda t: func_y(t)**2
        integral_y, _ = quad(integrand_y, t[0], t[-1], limit = INTLIM)
        print(f"Dataset #{i+1} batch #{j+1}:\t{np.sqrt(integral_x+integral_y):.5f}.")
        # quad_vec is actually slower than quad!
        # func = time_series_func_2D(t, x, y)
        # integrand = lambda t: func(t)**2
        # integral, _ = quad_vec(integrand, t[0], t[-1], limit = INTLIM)
        # print(f"Dataset #{i+1} batch #{j+1}: {np.sqrt(np.sum(integral)):.5f}.")
plt.show()