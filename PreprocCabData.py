import os
import numpy as np
import pandas as pd
from datetime import datetime

UNIT = 1000
EARTH_RADIUS = 6371*UNIT
SPEED_LIMIT = 180*UNIT/60/60

def convert_coord(lat, long):
    x = EARTH_RADIUS*np.deg2rad(long)
    y = EARTH_RADIUS*np.log(np.tan(0.25*np.pi+0.5*np.deg2rad(lat)))
    return float(x), float(y)

filename_list = []; x_list = []; y_list = []; t_list = []
for filename in sorted(os.listdir("cabspottingdata")):
    if filename != "_cabs.txt" and filename != "README" and filename != "trajectory.pkl":
        track = pd.read_csv("cabspottingdata/"+filename, header = None, sep = '\s+')
        filename_list.append(filename)
        print(f"Processing {filename}...")
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
df = pd.DataFrame({'filename': filename_list, 't': t_list, 'x': x_list, 'y': y_list})
df['x_min'] = df['x'].apply(min)
df['x_max'] = df['x'].apply(max)
df['y_min'] = df['y'].apply(min)
df['y_max'] = df['y'].apply(max)
df.to_pickle("cabspottingdata/trajectory.pkl")
