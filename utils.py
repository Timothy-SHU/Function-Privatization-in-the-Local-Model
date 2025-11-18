from os import makedirs
from os.path import isdir
import numpy as np
from glob import glob
from datetime import datetime


def deg_to_rad(deg):
    rad = deg/360.*2*np.pi
    return rad


def convert_coord(lat, long, long0=0, R=6371000.0):
    x = R*(long-long0)
    y = R*np.log(np.tan(0.25*np.pi+0.5*lat))
    return (x, y)


def Extract_Cab_Data(filename,b_clean=False,time_max=86400,sp_max=3000,R=6371000.0):
    data = np.genfromtxt(filename,names=None)
    trace = []
    ind_sorted = data[:,-1].argsort()
    
    if b_clean:
        n = len(ind_sorted)
        ind = ind_sorted[0]
        row = data[ind]
        prev_time = datetime.fromtimestamp(row[-1])
        (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
        trace.append((prev_time,x, y))
        prev_loc = (x, y)
        for i in range(1,n):
            ind = ind_sorted[i]
            row = data[ind]
            curr_time = datetime.fromtimestamp(row[-1])
            time_delta = (curr_time - prev_time).seconds
            if time_delta <= time_max:
                (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
                if np.sqrt((x-prev_loc[0])**2+(y-prev_loc[1])**2)/time_delta*60 <= sp_max:
                    trace.append((curr_time, x, y))
                    prev_time = curr_time
                    prev_loc = (x, y)
    else:
        for ind in ind_sorted:
            row = data[ind]
            curr_time = datetime.fromtimestamp(row[-1])
            (x, y) = convert_coord(deg_to_rad(row[0]),deg_to_rad(row[1]),R=R)
            trace.append((curr_time, x, y))
    return np.array(trace)


def Extract_Cab_Data_All(folder_in,len_min=5000,len_max=30000,R=6371000.0):
    dict_traj = {}
    trip_names = get_all_filenames(folder_in,'new_*.txt')
    for name in trip_names:
        x = Extract_Cab_Data(name,b_clean=True,R=R)
        n = len(x)
        if n < len_min or n > len_max:
            continue
        name_short = name.replace(folder_in,'').replace('/','').replace('\\','')
        dict_traj[name_short] = x
    return dict_traj


def get_all_filenames(dir,prefix=''):
    dirlist = glob(dir+'/'+prefix)
    return dirlist