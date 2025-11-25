import os
import numpy as np
import pandas as pd

repeat = 20

def getStats(filename, isBaseline = False, smoothed = False):
    file = open(filename, 'r')
    buffer = [float(x) for x in file.read().strip().split()]
    funcL2 = buffer[0]; buffer = buffer[1:]
    if not isBaseline:
        err_ls = buffer[0]
        approx_time = buffer[1]
        buffer = buffer[2:]
    res = []
    for i in range(repeat):
        if isBaseline:
            res.append([buffer[0], buffer[1]])
            buffer = buffer[2:]
        elif smoothed:
            res.append([buffer[1], buffer[0], buffer[2], 
                        buffer[4], buffer[3], buffer[5]])
            buffer = buffer[6:]
        else:
            res.append([buffer[1], buffer[0], buffer[2]])
            buffer = buffer[3:]
    res = sorted(res, key = lambda rec: rec[0])
    res = np.array(res[:-(repeat//10)])     # remove 10% outlier
    avg = res.mean(axis = 0)
    if isBaseline: return [funcL2] + avg.tolist()
    return [funcL2, err_ls, approx_time] + avg.tolist()

EPS = 0.01
GS = 400000
BATCH_SIZE = 1000
SVT_THRESHOLD = 10
NUM_POINTS = 11103894
SMOOTHED = True
dir = f"results/taxi_{EPS}_{BATCH_SIZE}_{SVT_THRESHOLD}/"
dir_bl = f"results/taxi_bl_{EPS}_{BATCH_SIZE}_{GS}/"
df = pd.DataFrame(columns = ['name', '# batches', 'total eps', 'func L2', 
                             'approx err', 'priv err', 'priv loss', 'smooth err', 'smooth loss', 
                             'approx time', 'priv time', 'smooth time', 'total runtime', 
                             'baseline priv err', 'baseline runtime'])
for idx, subdir in enumerate(os.scandir(dir)):
    if not subdir.is_dir(): continue
    folder = os.path.basename(subdir.path)+"/"
    funcL2 = None; stats_sum = np.zeros(10)
    for file in os.listdir(dir+folder):
        stats = getStats(dir+folder+file, False, SMOOTHED)
        stats_bl = getStats(dir_bl+folder+file, True)
        if funcL2 == None: funcL2 = stats[0]
        stats_sum += np.array(stats[1:]+stats_bl[1:])
    df.loc[idx] = {'name': folder[:-1], '# batches': len(os.listdir(dir+folder)), 
                   'total eps': len(os.listdir(dir+folder))*EPS, 
                   'func L2': funcL2, 'approx err': stats_sum[0], 
                   'priv err': stats_sum[2], 'priv loss': stats_sum[3], 
                   'smooth err': stats_sum[5], 'smooth loss': stats_sum[6], 
                   'approx time': stats_sum[1], 'priv time': stats_sum[4], 'smooth time': stats_sum[7], 
                   'total runtime': stats_sum[1]+stats_sum[4]+stats_sum[7], 
                   'baseline priv err': stats_sum[8], 'baseline runtime': stats_sum[9]}
df.to_csv("results/TaxiTrajectory.csv")
print("="*120)
print("Taxi Trajectory Dataset (units: t -- second, x/y -- meter)")
print(f"eps = {EPS}; batch size = {BATCH_SIZE}; SVT threshold = {SVT_THRESHOLD}.")
print("Below are average statistics per 1000 datapoints.")
print("-"*120)
print(" "*9+f"avg ||f|| = {df['func L2'].sum()/NUM_POINTS*BATCH_SIZE:>13.5f}.")
print(f"avg ||f-f_approx|| = {df['approx err'].sum()/NUM_POINTS*BATCH_SIZE:>13.5f};", end = " "*53)
print(f"avg approx time = {df['smooth time'].sum()/NUM_POINTS*BATCH_SIZE*1000:>9.5f} ms.")
print(f"  avg ||f-f_priv|| = {df['priv err'].sum()/NUM_POINTS*BATCH_SIZE:>13.5f};", end = "\t")
print(f"  avg ||f_priv-f_approx|| = {df['priv loss'].sum()/NUM_POINTS*BATCH_SIZE:>12.5f};", end = "\t")
print(f"  avg priv time = {df['priv time'].sum()/NUM_POINTS*BATCH_SIZE*1000:>9.5f} ms.")
print(f"avg ||f-f_smooth|| = {df['smooth err'].sum()/NUM_POINTS*BATCH_SIZE:>13.5f};", end = "\t")
print(f"avg ||f_smooth-f_approx|| = {df['smooth loss'].sum()/NUM_POINTS*BATCH_SIZE:>12.5f};", end = "\t")
print(f"avg smooth time = {df['smooth time'].sum()/NUM_POINTS*BATCH_SIZE*1000:>9.5f} ms.")
print(" "*86+f"avg total runtime = {df['total runtime'].sum()/NUM_POINTS*BATCH_SIZE*1000:>9.5f} ms.")
print("-"*120)
print(f"avg ||f-f_bl|| = {df['baseline priv err'].sum()/NUM_POINTS*BATCH_SIZE:.5f};", end = " "*48)
print(f"avg baseline runtime = {df['baseline runtime'].sum()/NUM_POINTS*BATCH_SIZE*1000:>9.5f} ms.")

EPS = 1.0
GS = 12000
BATCH_SIZE = 20
TIME_SCALE = 80
N = 1000
FREQUENCY = 100
NUM_RECORDS = 21799
SMOOTHED = False
dir = f"results/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}/"
dir_bl = f"results/ECG_bl_{EPS}_{GS}/"
df = pd.DataFrame(columns = ['folder', '# records', 'eps (per record)', 'avg func L2', 
                             'avg approx err', 'avg priv err', 'avg priv loss', 
                             'avg approx time', 'avg priv time', 'avg total runtime', 
                             'avg baseline priv err', 'avg baseline runtime'])
idx = 0; stats_total = np.zeros(8)
for i in range(22):
    folder = "{:05d}".format(i*1000)
    num_rec = 0; stats_avg = np.zeros(8)
    for j in range(1000):
        file = "{:05d}_lr.txt".format(i*1000+j)
        if not os.path.exists(dir+file): continue
        stats = getStats(dir+file, False, SMOOTHED)
        stats_bl = getStats(dir_bl+file, True)
        stats_avg += np.array(stats+stats_bl[1:])
        stats_total += np.array(stats+stats_bl[1:])
        num_rec += 1
    stats_avg /= num_rec
    df.loc[i] = {'folder': folder, '# records': num_rec, 'eps (per record)': EPS, 'avg func L2': stats_avg[0], 
                 'avg approx err': stats_avg[1], 'avg priv err': stats_avg[3], 'avg priv loss': stats_avg[4], 
                 'avg approx time': stats_avg[2], 'avg priv time': stats_avg[5], 
                 'avg total runtime': stats_avg[2]+stats_avg[5], 
                 'avg baseline priv err': stats_avg[6], 'avg baseline runtime': stats_avg[7]}
df.to_csv("results/ECG.csv")
print("="*120)
print("ECG Dataset (units: t -- 1/80 second, amp -- microvolt \u03BCv)")
print(f"eps = {EPS}; {N} points per record, sampled at frequency {FREQUENCY} Hz.")
print(f"Bounded sinc basis applied: {N//BATCH_SIZE} pieces x {BATCH_SIZE*TIME_SCALE//FREQUENCY} basis func per piece.")
print("Below are average statistics per record.")
print("-"*120)
print(" "*8+f"avg ||f|| = {stats_total[0]/NUM_RECORDS:10.5f}.")
print(f"avg ||f-f_approx|| = {stats_total[1]/NUM_RECORDS:>9.5f};", end = " "*41)
print(f"avg approx time = {stats_total[2]/NUM_RECORDS*1000:>9.5f} ms.")
print(f"  avg ||f-f_priv|| = {stats_total[3]/NUM_RECORDS:>9.5f};", end = "\t")
print(f"  avg ||f_priv-f_approx|| = {stats_total[4]/NUM_RECORDS:>9.5f};", end = "\t")
print(f"  avg priv time = {stats_total[5]/NUM_RECORDS*1000:>9.5f} ms.")
print(" "*70+f"avg total runtime = {(stats_total[2]+stats_total[5])/NUM_RECORDS*1000:>9.5f} ms.")
print("-"*120)
print(f" avg ||f-f_bl|| = {stats_total[6].sum()/NUM_RECORDS:.5f};", end = " "*36)
print(f"avg baseline runtime = {stats_total[7].sum()/NUM_RECORDS*1000:>9.5f} ms.")
print("="*120)