import os, sys, time, shutil
import numpy as np
import pandas as pd

timer = time.time()
df = pd.read_pickle("cabspottingdata/trajectory.pkl")
print("="*50)
print(f"Datasets loaded in {time.time()-timer:.2f} sec.")
print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
print("="*50)

new_t = []; new_x = []; new_y = []
for i in range(len(df)):
    start = 0; end = 0
    cnt_valid = 0; cnt_invalid = 0
    new_t.append([])
    new_x.append([])
    new_y.append([])
    while start < len(df['t'][i]):
        end = start
        while df['t'][i][start].date() == df['t'][i][end].date():
            if end+1 < len(df['t'][i]): end += 1
            else: break
        t = np.array(df['t'][i][start:end])
        l = t[0].replace(hour = 8, minute = 0, second = 0)
        r = t[0].replace(hour = 20, minute = 0, second = 0)
        if end-start <= 10 or t[0] > l or t[-1] < r:
            start = end+1; cnt_invalid += 1; continue
        x = np.array(df['x'][i][start:end])
        y = np.array(df['y'][i][start:end])
        lidx = np.searchsorted(t, l, side = 'left')
        ridx = np.searchsorted(t, r, side = 'right')-1
        if t[lidx] != l:
            l_sec = (l-t[lidx-1]).total_seconds()
            t_sec = [0, (t[lidx]-t[lidx-1]).total_seconds()]
            x[lidx-1] = np.interp(l_sec, t_sec, x[lidx-1:lidx+1])
            y[lidx-1] = np.interp(l_sec, t_sec, y[lidx-1:lidx+1])
            t[lidx-1] = l; lidx -= 1
        if t[ridx] != r:
            r_sec = (r-t[ridx]).total_seconds()
            t_sec = [0, (t[ridx+1]-t[ridx]).total_seconds()]
            x[ridx+1] = np.interp(r_sec, t_sec, x[ridx:ridx+2])
            y[ridx+1] = np.interp(r_sec, t_sec, y[ridx:ridx+2])
            t[ridx+1] = r; ridx += 1
        if ridx-lidx+1 < 500:
            start = end+1; cnt_invalid += 1; continue
        cnt_valid += 1
        t = t[lidx:ridx+1]; new_t[-1].append(t)
        x = x[lidx:ridx+1]; new_x[-1].append(x)
        y = y[lidx:ridx+1]; new_y[-1].append(y)
        start = end+1
    print(f"Dataset {df['filename'][i]}: selected {cnt_valid} out of {cnt_valid+cnt_invalid} curves.")

new_df = pd.DataFrame({'filename': df['filename'], 't': new_t, 'x': new_x, 'y': new_y})
new_df.to_pickle("cabspottingdata/trajectory_selected.pkl")