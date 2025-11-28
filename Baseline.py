import os, sys
import time, wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import laplace, gamma

bench = sys.argv[1]
EPS = float(sys.argv[2])
SAMPLE_RATE = float(sys.argv[3])
WINDOW_SCALE = float(sys.argv[4])
TIME_SCALE = 80
VAL_SCALE = 1000
UNIT_TIME_SCALE = 43200
repeat = 20

def sqrInt(t, val1, val2):
    ret = 0
    for i in range(len(t)-1):
        l = t[i]; r = t[i+1]
        if l >= r: continue
        k1 = (val1[i+1]-val1[i])/(r-l); b1 = val1[i]-k1*l
        k2 = (val2[i+1]-val2[i])/(r-l); b2 = val2[i]-k2*l
        k = k1-k2; b = b1-b2
        # integrate ((k1-k2)t+(b1-b2))^2 = (kt+b)^2 over [l,r]
        # \int k^2t^2+2kbt+b^2 = 1/3*k^2t^3+kbt^2+b^2*t
        ret += 1/3*(k**2)*(r**3)+k*b*(r**2)+(b**2)*r
        ret -= 1/3*(k**2)*(l**3)+k*b*(l**2)+(b**2)*l
    return ret

def Lap(scale, dim = 1):
    if dim == 1:
        return laplace.rvs(scale = scale)
    noise = np.random.randn(dim)
    noise = noise/np.linalg.norm(noise)
    noise *= gamma.rvs(a = dim, scale = scale)
    return noise

if bench in ['t', 'T', 'taxi', 'Taxi']:
    df = pd.read_pickle("cabspottingdata/trajectory_selected.pkl")
    for i in tqdm(range(len(df))):
        for j in range(len(df['t'][i])):
            iter_timer = time.time()
            t = df['t'][i][j]
            t = np.array([(cur-t[0]).total_seconds() for cur in t])
            eps = EPS/UNIT_TIME_SCALE*(t[-1]-t[0])
            x = np.array(df['x'][i][j])
            y = np.array(df['y'][i][j])
            min_x = np.min(x); x -= min_x
            min_y = np.min(y); y -= min_y

            SAMPLE = max(int(len(t)*SAMPLE_RATE), 2)
            WINDOW = max(1, int(SAMPLE*WINDOW_SCALE/2))

            filename = df['filename'][i].removeprefix("new_").removesuffix(".txt")
            dir = f"results/TaxiTrajectory/taxi_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/{filename}/"
            os.makedirs(dir, exist_ok = True)
            res_file = open(dir+f"{filename}-{j+1}.txt", 'w')
            res_file.write(f"{SAMPLE} {WINDOW*2+1}\n")
            funcSqrInt = sqrInt(t, x, np.zeros(len(t)))
            funcSqrInt += sqrInt(t, y, np.zeros(len(t)))
            res_file.write(f"{np.sqrt(funcSqrInt)}\n\n")

            for k in range(repeat):
                priv_timer = time.time()
                sample = np.linspace(0, len(t)-1, SAMPLE, dtype = int)
                sample = [i in sample for i in range(len(t))]
                x_priv = x[sample]; y_priv = y[sample]
                for l in range(SAMPLE):
                    noise = Lap(SAMPLE/EPS, 2)
                    x_priv[l] += noise[0]
                    y_priv[l] += noise[1]
                x_priv_pts = np.interp(t, t[sample], x_priv)
                y_priv_pts = np.interp(t, t[sample], y_priv)
                priv_time = time.time()-priv_timer

                smooth_timer = time.time()
                x_smooth = np.zeros(SAMPLE)
                y_smooth = np.zeros(SAMPLE)
                for l in range(SAMPLE):
                    x_smooth[l] = np.mean(x_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
                    y_smooth[l] = np.mean(y_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
                x_smooth_pts = np.interp(t, t[sample], x_smooth)
                y_smooth_pts = np.interp(t, t[sample], y_smooth)
                smooth_time = time.time()-smooth_timer

                err_priv = sqrInt(t, x, x_priv_pts)
                err_priv += sqrInt(t, y, y_priv_pts)
                err_priv = np.sqrt(err_priv)
                res_file.write(f"{err_priv} {priv_time}\n")
                err_smooth = sqrInt(t, x, x_smooth_pts)
                err_smooth += sqrInt(t, y, y_smooth_pts)
                err_smooth = np.sqrt(err_smooth)
                res_file.write(f"{err_smooth} {smooth_time}\n\n")
            res_file.close()
        # if i == 2: break   # sample: run only the first three datasets

elif bench in ['e', 'E', 'ecg', 'ECG']:
    records = []
    min_val = 0; max_val = 0
    for i in range(1, 21838):
    # for i in range (1, 21):  # sample: run only the first 20 records
        folder = "{:05d}".format(i//1000*1000)
        file = "{:05d}_lr".format(i)
        path = "ptb-xl/records100/"+folder+"/"+file
        try:
            record = wfdb.rdrecord(path)
            records.append((folder, file, record))
            min_val = min(min_val, np.min(record.p_signal[:, 1])*VAL_SCALE)
            max_val = max(max_val, np.max(record.p_signal[:, 1])*VAL_SCALE)
        except:
            pass

    for folder, file, record in tqdm(records, position = 0, leave = True):
        iter_timer = time.time()
        n = record.p_signal.shape[0]
        T = n//record.fs*TIME_SCALE
        t = np.linspace(1/record.fs*TIME_SCALE, T, n)
        val = record.p_signal[:, 1]*VAL_SCALE

        SAMPLE = int(len(t)*SAMPLE_RATE)
        WINDOW = max(1, int(SAMPLE*WINDOW_SCALE/2))

        dir = f"results/ECG/ECG_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/{folder}/"
        os.makedirs(dir, exist_ok = True)
        res_file = open(dir+f"{file}.txt", 'w')
        res_file.write(f"{SAMPLE} {WINDOW*2+1}\n")
        funcSqrInt = sqrInt(t, val, np.zeros(len(t)))
        res_file.write(f"{np.sqrt(funcSqrInt)}\n\n")
        for k in range(repeat):
            priv_timer = time.time()
            sample = np.linspace(0, n-1, SAMPLE, dtype = int)
            sample = [i in sample for i in range(len(t))]
            val_priv = val[sample]
            for i in range(SAMPLE):
                val_priv[i] += Lap(SAMPLE/EPS)
            val_priv_pts = np.interp(t, t[sample], val_priv)
            priv_time = time.time()-priv_timer

            smooth_timer = time.time()
            val_smooth = np.zeros(SAMPLE)
            for l in range(SAMPLE):
                val_smooth[l] = np.mean(val_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
            val_smooth_pts = np.interp(t, t[sample], val_smooth)
            smooth_time = time.time()-smooth_timer

            err_priv = np.sqrt(sqrInt(t, val, val_priv_pts))
            res_file.write(f"{err_priv} {priv_time}\n")
            err_smooth = np.sqrt(sqrInt(t, val, val_smooth_pts))
            res_file.write(f"{err_smooth} {smooth_time}\n\n")
        res_file.close()

else:
    print(f"No such benchmark {bench}!")