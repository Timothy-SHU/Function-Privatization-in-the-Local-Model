import os, sys
import numpy as np
import pandas as pd

N = 1000
FREQUENCY = 100
TIME_SCALE = 80
UNIT_TIME_SCALE = 43200
repeat = 20

def getStats(filename, isBaseline = False, smoothed = False):
    file = open(filename, 'r')
    buffer = [float(x) for x in file.read().strip().split()]
    meta = buffer[:2]; buffer = buffer[2:]
    funcL2 = buffer[0]; buffer = buffer[1:]
    if not isBaseline:
        err_ls = buffer[0]
        approx_time = buffer[1]
        buffer = buffer[2:]
    res = []
    for i in range(repeat):
        if isBaseline:
            res.append([buffer[0], buffer[1], 
                        buffer[2], buffer[3]])
            buffer = buffer[4:]
        elif smoothed:
            res.append([buffer[1], buffer[0], buffer[2], 
                        buffer[4], buffer[3], buffer[5]])
            buffer = buffer[6:]
        else:
            res.append([buffer[1], buffer[0], buffer[2]])
            buffer = buffer[3:]
    res = sorted(res, key = lambda rec: rec[0])
    res = np.array(res[:-(repeat//10)])     # remove 10% outliers
    avg = res.mean(axis = 0)
    if isBaseline: return [funcL2] + meta + avg.tolist()
    return [funcL2] + meta + [err_ls, approx_time] + avg.tolist()

def getTaxiRes(EPS, SAMPLE_RATE, WINDOW_SCALE, SMOOTHED):
    dir = f"results/TaxiTrajectory/taxi_{EPS}/"
    dir_bl = f"results/TaxiTrajectory/taxi_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    df = pd.DataFrame(columns = ['name', 'total length of time', 'total eps', 'func L2', 
                                'approx err', 'priv err', 'priv loss', 'smooth err', 'smooth loss', 
                                'approx time', 'priv time', 'smooth time', 'total runtime', 
                                'baseline priv err', 'baseline runtime', 
                                'baseline smooth err', 'baseline smooth time', 
                                'baseline total runtime'])
    for idx, subdir in enumerate(sorted(os.scandir(dir), key = lambda e: e.name)):
        if not subdir.is_dir(): continue
        folder = os.path.basename(subdir.path)+"/"
        funcL2 = None; stats_sum = np.zeros(14)
        for file in os.listdir(dir+folder):
            stats = getStats(dir+folder+file, False, SMOOTHED)
            stats_bl = getStats(dir_bl+folder+file, True)
            if funcL2 == None: funcL2 = stats[0]
            stats_sum += np.array(stats[1:]+stats_bl[3:])
        df.loc[idx] = {'name': folder[:-1], 
                       'total length of time': stats_sum[0], 
                       'total eps': stats_sum[1], 
                       'func L2': funcL2, 'approx err': stats_sum[2], 
                       'priv err': stats_sum[4], 'priv loss': stats_sum[5], 
                       'smooth err': stats_sum[7], 'smooth loss': stats_sum[8], 
                       'approx time': stats_sum[3], 'priv time': stats_sum[6], 'smooth time': stats_sum[9], 
                       'total runtime': stats_sum[3]+stats_sum[6]+stats_sum[9], 
                       'baseline priv err': stats_sum[10], 'baseline runtime': stats_sum[11], 
                       'baseline smooth err': stats_sum[12], 'baseline smooth time': stats_sum[13], 
                       'baseline total runtime': stats_sum[11]+stats_sum[13]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/TaxiTrajectory_{EPS}.csv")
    print(f"Privacy budget: {EPS} per 12h;", end = " ")
    print(f"baseline sample size = {SAMPLE_RATE} x num datapoints;", end = " ")
    print(f"baseline window size = {WINDOW_SCALE} x sample size.")
    print("-"*120)
    total_units = df['total length of time'].sum()/UNIT_TIME_SCALE
    print(f'''{f"avg ||f|| = {df['func L2'].sum()/total_units:>14.5f}.":>36}''')
    print(f'''{f"avg ||f-f_approx|| = {df['approx err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg approx time = {df['smooth time'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg ||f-f_priv|| = {df['priv err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg ||f_priv-f_approx|| = {df['priv loss'].sum()/total_units:>12.5f};":>46}''', end = "")
    print(f'''{f"avg priv time = {df['priv time'].sum()/total_units*1000:>8.5f} ms.":>38}''')
    print(f'''{f"avg ||f-f_smooth|| = {df['smooth err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg ||f_smooth-f_approx|| = {df['smooth loss'].sum()/total_units:>12.5f};":>46}''', end = "")
    print(f'''{f"avg smooth time = {df['smooth time'].sum()/total_units*1000:>8.5f} ms.":>38}''')
    print(f'''{f"avg total runtime = {df['total runtime'].sum()/total_units*1000:>8.5f} ms.":>120}''')
    print("-"*120)
    print(f'''{f"avg ||f-f_bl|| = {df['baseline priv err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg baseline runtime = {df['baseline runtime'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg ||f-f_bl_sm|| = {df['baseline smooth err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg baseline smooth time = {df['baseline smooth time'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg baseline total runtime = {df['baseline total runtime'].sum()/total_units*1000:>8.5f} ms.":>120}''')
    print("="*120)

def getECGRes(EPS, BATCH_SIZE, SAMPLE_RATE, WINDOW_SCALE, SMOOTHED):
    dir = f"results/ECG/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}/"
    dir_bl = f"results/ECG/ECG_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    df = pd.DataFrame(columns = ['folder', '# records', 'eps (per record)', 'avg func L2', 
                                'avg approx err', 'avg priv err', 'avg priv loss', 
                                'avg approx time', 'avg priv time', 'avg total runtime', 
                                'avg baseline priv err', 'avg baseline runtime', 
                                'avg baseline smooth err', 'avg baseline smooth time', 
                                'avg baseline total runtime'])
    stats_total = np.zeros(10); total_rec = 0
    for i in range(22):
        folder = "{:05d}/".format(i*1000)
        num_rec = 0; stats_avg = np.zeros(10)
        for j in range(1000):
            file = "{:05d}_lr.txt".format(i*1000+j)
            if not os.path.exists(dir+folder+file): continue
            stats = getStats(dir+folder+file, False, SMOOTHED)
            stats_bl = getStats(dir_bl+folder+file, True)
            stats_avg += np.array([stats[0]]+stats[3:]+stats_bl[3:])
            stats_total += np.array([stats[0]]+stats[3:]+stats_bl[3:])
            num_rec += 1
        stats_avg /= num_rec
        total_rec += num_rec
        df.loc[i] = {'folder': folder[:-1], '# records': num_rec, 'eps (per record)': EPS, 'avg func L2': stats_avg[0], 
                    'avg approx err': stats_avg[1], 'avg priv err': stats_avg[3], 'avg priv loss': stats_avg[4], 
                    'avg approx time': stats_avg[2], 'avg priv time': stats_avg[5], 
                    'avg total runtime': stats_avg[2]+stats_avg[5], 
                    'avg baseline priv err': stats_avg[6], 'avg baseline runtime': stats_avg[7], 
                    'avg baseline smooth err': stats_avg[8], 'avg baseline smooth time': stats_avg[9], 
                    'avg baseline total runtime': stats_avg[7]+stats_avg[9]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}.csv")
    print(f"Privacy budget per record: {EPS};", end = " ")
    print(f"bounded sinc basis applied: {N//BATCH_SIZE} pieces x {BATCH_SIZE*TIME_SCALE//FREQUENCY} basis func per piece.")
    print("-"*120)
    print(f'''{f"avg ||f|| = {stats_total[0]/total_rec:10.5f}.":>32}''')
    print(f'''{f"avg ||f-f_approx|| = {stats_total[1]/total_rec:>10.5f};":>32}''', end = "")
    print(f'''{f"avg approx time = {stats_total[2]/total_rec*1000:>9.5f} ms.":>88}''')
    print(f'''{f"avg ||f-f_priv|| = {stats_total[3]/total_rec:>10.5f};":>32}''', end = "")
    print(f'''{f"avg ||f_priv-f_approx|| = {stats_total[4]/total_rec:>10.5f};":>44}''', end = "")
    print(f'''{f"avg priv time = {stats_total[5]/total_rec*1000:>9.5f} ms.":>44}''')
    print(f'''{f"avg total runtime = {(stats_total[2]+stats_total[5])/total_rec*1000:>9.5f} ms.":>120}''')
    print("-"*120)
    print(f'''{f"avg ||f-f_bl|| = {stats_total[6].sum()/total_rec:>10.5f};":>32}''', end = "")
    print(f'''{f"avg baseline runtime = {stats_total[7].sum()/total_rec*1000:>9.5f} ms.":>88}''')
    print(f'''{f"avg ||f-f_bl_sm|| = {stats_total[8].sum()/total_rec:>10.5f};":>32}''', end = "")
    print(f'''{f"avg baseline smooth time = {stats_total[9].sum()/total_rec*1000:>9.5f} ms.":>88}''')
    print(f'''{f"avg baseline total runtime = {(stats_total[7]+stats_total[9])/total_rec*1000:>9.5f} ms.":>120}''')
    print("="*120)

if sys.argv[1] in ["t", "T", "taxi", "Taxi"]:
    print("="*120)
    print("Taxi Trajectory Dataset (units: t -- second, x/y -- meter)")
    print("Below are average statistics per 1000 datapoints.")
    print("="*120)
    for EPS in [0.001, 0.01, 0.1, 1.0]:
        getTaxiRes(EPS, 0.1, 0.05, True)
elif sys.argv[1] in ["e", "E", "ecg", "ECG"]:
    print("="*120)
    print(f"ECG Dataset (units: t -- 1/{TIME_SCALE} second, amp -- microvolt \u03BCv)")
    print(f"{N} points per record, sampled at frequency {FREQUENCY} Hz.")
    print("Below are average statistics per record.")
    print("="*120)
    for EPS in [0.5, 1.0, 2.0, 4.0]:
        getECGRes(EPS, 20, 0.1, 0.05, False)