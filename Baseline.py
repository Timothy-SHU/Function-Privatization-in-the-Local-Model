import os, sys
import time, wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import laplace, gamma

bench = sys.argv[1]
EPS = float(sys.argv[2])
GS = int(sys.argv[3])
BATCH_SIZE = 1000
TIME_SCALE = 80
VAL_SCALE = 1000
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
    timer = time.time()
    df = pd.read_pickle("cabspottingdata/trajectory.pkl")
    print("="*90)
    print(f"Datasets loaded in {time.time()-timer:.2f} sec.")
    print(f"Total # of datapoints: {np.sum([len(df['t'][i]) for i in range(len(df))])}")
    x_min = np.min(df['x'].apply(np.min))
    x_max = np.min(df['x'].apply(np.max))
    y_min = np.min(df['y'].apply(np.min))
    y_max = np.min(df['y'].apply(np.max))
    print(f"x_min = {x_min}; x_max = {x_max}; range = {x_max-x_min}.")
    print(f"y_min = {y_min}; y_max = {y_max}; range = {y_max-y_min}.")
    print(f"Min L_infty GS required: {max(x_max-x_min, y_max-y_min)}.")
    print("="*90)

    timer = time.time()
    for i in tqdm(range(len(df))):
        for j in tqdm(range((len(df['t'][i])-1)//BATCH_SIZE+1), leave = False):
            iter_timer = time.time()
            t = df['t'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
            t = [(cur-t[0]).total_seconds() for cur in t]
            x = df['x'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
            y = df['y'][i][j*BATCH_SIZE : min((j+1)*BATCH_SIZE, len(df['t'][i]))]
            min_x = np.min(x); x -= min_x
            min_y = np.min(y); y -= min_y

            filename = df['filename'][i].removeprefix("new_").removesuffix(".txt")
            dir = f"results/taxi_bl_{EPS}_{BATCH_SIZE}_{GS}/{filename}/"
            os.makedirs(dir, exist_ok = True)
            res_file = open(dir+f"{filename}-{j+1}.txt", 'w')
            funcSqrInt = sqrInt(t, x, np.zeros(len(t)))
            funcSqrInt += sqrInt(t, y, np.zeros(len(t)))
            res_file.write(f"{np.sqrt(funcSqrInt)}\n")
            for k in range(repeat):
                priv_timer = time.time()
                x_priv = np.zeros(len(t))
                y_priv = np.zeros(len(t))
                for l in range(len(t)):
                    # L_infty norm, independent Lap noises
                    x_priv[l] = x[l]+Lap(GS/EPS)
                    y_priv[l] = y[l]+Lap(GS/EPS)
                priv_time = time.time()-priv_timer
                err_priv = sqrInt(t, x, x_priv)
                err_priv += sqrInt(t, y, y_priv)
                err_priv = np.sqrt(err_priv)
                res_file.write(f"{err_priv} {priv_time}\n")
            res_file.close()
        # break   # sample: run only the first dataset
    print(f"Total time elapsed: {time.time()-timer:.2f} sec.")

elif bench in ['e', 'E', 'ecg', 'ECG']:
    records = []
    timer = time.time()
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
    print("="*50)
    print(f"{len(records)} records loaded in {time.time()-timer:.2f} sec.")
    print(f"Each {records[0][2].p_signal.shape[0]} datapoints, sampled at frequency {records[0][2].fs} Hz.")
    print(f"min_val = {min_val}; max_val = {max_val}.")
    print(f"Min L_infty GS required: {max_val-min_val}.")
    print("="*50)

    timer = time.time()
    for folder, file, record in tqdm(records, position = 0, leave = True):
        iter_timer = time.time()
        n = record.p_signal.shape[0]
        T = n//record.fs*TIME_SCALE
        t = np.linspace(1/record.fs*TIME_SCALE, T, n)
        val = record.p_signal[:, 1]*VAL_SCALE

        dir = f"results/ECG_bl_{EPS}_{GS}/"
        os.makedirs(dir, exist_ok = True)
        res_file = open(dir+f"{file}.txt", 'w')
        funcSqrInt = sqrInt(t, val, np.zeros(len(t)))
        res_file.write(f"{np.sqrt(funcSqrInt)}\n")
        for k in range(repeat):
            priv_timer = time.time()
            val_priv = np.zeros(n)
            for i in range(n):
                val_priv[i] = val[i]+Lap(GS/EPS)
            priv_time = time.time()-priv_timer
            err_priv = np.sqrt(sqrInt(t, val, val_priv))
            res_file.write(f"{err_priv} {priv_time}\n")
        res_file.close()
    print(f"Total time elapsed: {time.time()-timer:.2f} sec.")

else:
    print(f"No such benchmark {bench}!")