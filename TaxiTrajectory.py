from PrivPwcApprox import *
from AdaptApprox import *
from datetime import datetime

UNIT = 1000
EARTH_RADIUS = 6371*UNIT
SPEED_LIMIT = 180*UNIT/60/60

BATCH_SIZE = 100
SVT_THRESHOLD_SCALE = 100

def convert_coord(lat, long):
    x = EARTH_RADIUS*np.deg2rad(long)
    y = EARTH_RADIUS*np.log(np.tan(0.25*np.pi+0.5*np.deg2rad(lat)))
    return x, y

timer = time.time()
filename_list = []; x_list = []; y_list = []; t_list = []
for filename in sorted(os.listdir("cabspottingdata")):
    if filename != "_cabs.txt" and filename != "README":
        track = pd.read_csv("cabspottingdata/"+filename, header = None, sep = '\s+')
        filename_list.append(filename)
        logging.info(f"Processing {filename}...")
        prev_time = None
        for i in reversed(range(len(track))):
            lat = track.iloc[i, 0]
            long = track.iloc[i, 1]
            x, y = convert_coord(lat, long)
            cur_time = datetime.fromtimestamp(track.iloc[i, 3])
            if prev_time == None:
                x_list.append([x])
                y_list.append([y])
                t_list.append([0])
                prev_time = cur_time
            else:
                delta_x = x-x_list[-1][-1]
                delta_y = y-y_list[-1][-1]
                delta_t = (cur_time-prev_time).seconds
                if np.sqrt(delta_x**2+delta_y**2) <= delta_t*SPEED_LIMIT:
                    x_list[-1].append(x)
                    y_list[-1].append(y)
                    t_list[-1].append(t_list[-1][-1]+delta_t)
                    prev_time = cur_time
        break
df = pd.DataFrame({'filename': filename_list, 't': t_list, 'x': x_list, 'y': y_list})
df['x_min'] = df['x'].apply(np.min)
df['x_max'] = df['x'].apply(np.max)
df['y_min'] = df['y'].apply(np.min)
df['y_max'] = df['y'].apply(np.max)
logging.info(df)
print("="*50)
print(f"Datasets loaded in {time.time()-timer:.5f} sec.")
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)

timer = time.time()
iter_timer = time.time()
for i in range(len(df)):
    # for j in range(len(df['t'][i])//BATCH_SIZE):
    for j in range(1):
        t = df['t'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        x = df['x'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y = df['y'][i][j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        min_x = np.min(x); x -= min_x
        min_y = np.min(y); y -= min_y
        func_x = time_series_func(t, x)
        func_y = time_series_func(t, y)
        approx_x, priv_x = adaptive_poly_approx(func = func_x, interval = (t[0], t[-1]), degree = 1, 
                                                SVT_threshold_scale = SVT_THRESHOLD_SCALE)
        approx_y, priv_y = adaptive_poly_approx(func = func_y, interval = (t[0], t[-1]), degree = 1, 
                                                SVT_threshold_scale = SVT_THRESHOLD_SCALE)
        # logging.info(f"Approx x: {approx_x(t)}\nApprox y: {approx_y(t)}\n\n")
        # print(l2_dist(func_x, approx_x, t[0], t[-1]), l2_dist(func_x, priv_x, t[0], t[-1]))
        # print(l2_dist(func_y, approx_y, t[0], t[-1]), l2_dist(func_y, priv_y, t[0], t[-1]))
        plt.plot(x+min_x, y+min_y, color = 'black')
        plt.plot(approx_x(t)+min_x, approx_y(t)+min_y, color = 'red')
        plt.plot(priv_x(t)+min_x, priv_y(t)+min_y, color = 'blue')
        print(f"Dataset #{i+1} batch #{j+1} done. Executed in {time.time()-iter_timer:.5f} sec.")
        iter_timer = time.time()
print(f"Total time elapsed: {time.time()-timer:.5f} sec.")
plt.show()
