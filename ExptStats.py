import os, sys
import numpy as np
import pandas as pd

N = 1000
FREQUENCY = 100
TIME_SCALE = 80
UNIT_TIME_SCALE = 43200
repeat = 10

def getStats(filename, isBaseline = False, adaptive = False, smoothed = True):
    file = open(filename, 'r')
    buffer = [float(x) for x in file.read().strip().split()]
    meta = buffer[:2]; buffer = buffer[2:]
    funcL2 = buffer[0]; buffer = buffer[1:]
    if (not isBaseline) and (not adaptive):
        err_ls = buffer[0]
        approx_time = buffer[1]
        buffer = buffer[2:]
    res = []
    for i in range(repeat):
        rec = []
        if adaptive:
            rec += [buffer[0], buffer[1]]
            buffer = buffer[2:]
        if isBaseline:
            rec += [buffer[0], buffer[1], 
                    buffer[2], buffer[3]]
            buffer = buffer[4:]
        elif smoothed:
            rec += [buffer[1], buffer[0], buffer[2], 
                    buffer[4], buffer[3], buffer[5]]
            buffer = buffer[6:]
        else:
            rec += [buffer[1], buffer[0], buffer[2]]
            buffer = buffer[3:]
        res.append(rec)
    if adaptive:
        res = sorted(res, key = lambda rec: rec[2])
    else:
        res = sorted(res, key = lambda rec: rec[0])
    res = np.array(res[(repeat//10):-(repeat//10)])     # remove 20% outliers
    avg = res.mean(axis = 0)
    if isBaseline or adaptive: return [funcL2] + meta + avg.tolist()
    return [funcL2] + meta + [err_ls, approx_time] + avg.tolist()

def getTaxiRes(METHOD, EPS):
    dir = f"results/TaxiTrajectory/taxi_{EPS}/"
    df = pd.DataFrame(columns = ['name', 'total length of time', 'total eps', 'func L2', 
                                 'approx err', 'priv err', 'priv loss', 'smooth err', 'smooth loss', 
                                 'approx time', 'priv time', 'smooth time', 'total runtime'])
    for idx, subdir in enumerate(sorted(os.scandir(dir), key = lambda e: e.name)):
        if not subdir.is_dir(): continue
        folder = os.path.basename(subdir.path)+"/"
        funcL2 = None; stats_sum = np.zeros(10)
        for file in os.listdir(dir+folder):
            if METHOD not in file: continue
            stats = getStats(dir+folder+file, False, True, True)
            if funcL2 == None: funcL2 = stats[0]
            stats_sum += np.array(stats[1:])
        df.loc[idx] = {'name': folder[:-1], 'total length of time': stats_sum[0], 
                       'total eps': stats_sum[1], 'func L2': funcL2, 'approx err': stats_sum[2], 
                       'priv err': stats_sum[4], 'priv loss': stats_sum[5], 
                       'smooth err': stats_sum[7], 'smooth loss': stats_sum[8], 
                       'approx time': stats_sum[3], 'priv time': stats_sum[6], 'smooth time': stats_sum[9], 
                       'total runtime': stats_sum[3]+stats_sum[6]+stats_sum[9]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/TaxiTrajectory_{EPS}.csv")
    total_units = df['total length of time'].sum()/UNIT_TIME_SCALE
    print("="*120)
    print("Taxi Trajectory Dataset (units: t -- second, x/y -- meter)")
    print(f"Privacy budget: {EPS} per curve; {int(total_units)} curves in total.")
    print("Below are average statistics per curve (12h time range).")
    print("-"*120)
    total_units = df['total length of time'].sum()/UNIT_TIME_SCALE
    print(f'''{f"avg ||f|| = {df['func L2'].sum()/total_units:>14.5f}.":>36}''')
    print(f'''{f"avg ||f-f_approx|| = {df['approx err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg approx time = {df['approx time'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg ||f-f_priv|| = {df['priv err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg ||f_priv-f_approx|| = {df['priv loss'].sum()/total_units:>12.5f};":>46}''', end = "")
    print(f'''{f"avg priv time = {df['priv time'].sum()/total_units*1000:>8.5f} ms.":>38}''')
    print(f'''{f"avg ||f-f_smooth|| = {df['smooth err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg ||f_smooth-f_approx|| = {df['smooth loss'].sum()/total_units:>12.5f};":>46}''', end = "")
    print(f'''{f"avg smooth time = {df['smooth time'].sum()/total_units*1000:>8.5f} ms.":>38}''')
    print(f'''{f"avg total runtime = {df['total runtime'].sum()/total_units*1000:>8.5f} ms.":>120}''')

def getTaxiBLRes(METHOD, EPS, SAMPLE_RATE, WINDOW_SCALE):
    dir = f"results/TaxiTrajectory/taxi_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    df = pd.DataFrame(columns = ['name', 'func L2', 'priv err', 'priv time', 'smooth err', 'smooth time', 'total runtime'])
    total_units = 0
    for idx, subdir in enumerate(sorted(os.scandir(dir), key = lambda e: e.name)):
        if not subdir.is_dir(): continue
        folder = os.path.basename(subdir.path)+"/"
        funcL2 = None; stats_sum = np.zeros(4)
        for file in os.listdir(dir+folder):
            if METHOD not in file: continue
            stats = getStats(dir+folder+file, True)
            if funcL2 == None: funcL2 = stats[0]
            stats_sum += np.array(stats[3:])
            total_units += 1
        df.loc[idx] = {'name': folder[:-1], 'func L2': funcL2, 
                       'priv err': stats_sum[0], 'priv time': stats_sum[1], 
                       'smooth err': stats_sum[2], 'smooth time': stats_sum[3], 
                       'total runtime': stats_sum[1]+stats_sum[3]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/TaxiTrajectory_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}.csv")
    print("-"*120)
    print("Baseline:", end = " ")
    print(f"sample size = {SAMPLE_RATE} x num datapoints;", end = " ")
    print(f"window size = {WINDOW_SCALE} x sample size.")
    print(f'''{f"avg ||f-f_bl|| = {df['priv err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg baseline runtime = {df['priv time'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg ||f-f_bl_sm|| = {df['smooth err'].sum()/total_units:>14.5f};":>36}''', end = "")
    print(f'''{f"avg baseline smooth time = {df['smooth time'].sum()/total_units*1000:>8.5f} ms.":>84}''')
    print(f'''{f"avg baseline total runtime = {df['total runtime'].sum()/total_units*1000:>8.5f} ms.":>120}''')

def getECGRes(METHOD, EPS, BATCH_SIZE):
    dir = f"results/ECG/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}/"
    df = pd.DataFrame(columns = ['folder', '# records', 'eps (per record)', 
                                 'avg func L2', 'avg approx err', 
                                 'avg priv err', 'avg priv loss', 
                                 'avg smooth err', 'avg smooth loss', 
                                 'avg approx time', 'avg priv time', 
                                 'avg smooth time', 'avg total runtime'])
    stats_total = np.zeros(9); total_rec = 0
    # for i in range(22):
    for i in range(1):
        folder = "{:05d}/".format(i*1000)
        num_rec = 0; stats_avg = np.zeros(9)
        for j in range(1000):
            file = f"{i*1000+j:05d}_lr_{METHOD}.txt"
            if not os.path.exists(dir+folder+file): continue
            stats = getStats(dir+folder+file, False, False, True)
            stats_avg += np.array([stats[0]]+stats[3:])
            stats_total += np.array([stats[0]]+stats[3:])
            num_rec += 1
        stats_avg /= num_rec
        total_rec += num_rec
        df.loc[i] = {'folder': folder[:-1], '# records': num_rec, 'eps (per record)': EPS, 
                     'avg func L2': stats_avg[0], 'avg approx err': stats_avg[1], 
                     'avg priv err': stats_avg[3], 'avg priv loss': stats_avg[4], 
                     'avg smooth err': stats_avg[6], 'avg smooth loss': stats_avg[7], 
                     'avg approx time': stats_avg[2], 'avg priv time': stats_avg[5], 
                     'avg smooth_time': stats_avg[8], 
                     'avg total runtime': stats_avg[2]+stats_avg[5]+stats_avg[8]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//FREQUENCY}x{N//BATCH_SIZE}.csv")
    print("="*120)
    print(f"ECG Dataset (units: t -- 1/{TIME_SCALE} second, amp -- microvolt \u03BCv)")
    print(f"{total_rec} records in total: {N} points per record, sampled at frequency {FREQUENCY} Hz.")
    print(f"Privacy budget per record: {EPS};", end = " ")
    print(f"bounded sinc basis applied: {N//BATCH_SIZE} pieces x {BATCH_SIZE*TIME_SCALE//FREQUENCY} basis func per piece.")
    print("Below are average statistics per record.")
    print("-"*120)
    print(f'''{f"avg ||f|| = {stats_total[0]/total_rec:11.5f}.":>33}''')
    print(f'''{f"avg ||f-f_approx|| = {stats_total[1]/total_rec:>11.5f};":>33}''', end = "")
    print(f'''{f"avg approx time = {stats_total[2]/total_rec*1000:>9.5f} ms.":>87}''')
    print(f'''{f"avg ||f-f_priv|| = {stats_total[3]/total_rec:>11.5f};":>33}''', end = "")
    print(f'''{f"avg ||f_priv-f_approx|| = {stats_total[4]/total_rec:>11.5f};":>43}''', end = "")
    print(f'''{f"avg priv time = {stats_total[5]/total_rec*1000:>9.5f} ms.":>44}''')
    print(f'''{f"avg ||f-f_smooth|| = {stats_total[6]/total_rec:>11.5f};":>33}''', end = "")
    print(f'''{f"avg ||f_smooth-f_approx|| = {stats_total[7]/total_rec:>11.5f};":>43}''', end = "")
    print(f'''{f"avg smooth time = {stats_total[8]/total_rec*1000:>9.5f} ms.":>44}''')
    print(f'''{f"avg total runtime = {(stats_total[2]+stats_total[5]+stats_total[8])/total_rec*1000:>9.5f} ms.":>120}''')

def getECGBLRes(METHOD, EPS, SAMPLE_RATE, WINDOW_SCALE):
    dir = f"results/ECG/ECG_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}/"
    df = pd.DataFrame(columns = ['folder', '# records', 'eps (per record)', 
                                 'avg func L2', 'avg priv err', 'avg priv time', 
                                 'avg smooth err', 'avg smooth time', 'avg total runtime'])
    stats_total = np.zeros(5); total_rec = 0
    # for i in range(22):
    for i in range(1):
        folder = "{:05d}/".format(i*1000)
        num_rec = 0; stats_avg = np.zeros(5)
        for j in range(1000):
            file = f"{i*1000+j:05d}_lr_{METHOD}.txt"
            if not os.path.exists(dir+folder+file): continue
            stats = getStats(dir+folder+file, True)
            stats_avg += np.array([stats[0]]+stats[3:])
            stats_total += np.array([stats[0]]+stats[3:])
            num_rec += 1
        if num_rec > 0:
            stats_avg /= num_rec
        total_rec += num_rec
        df.loc[i] = {'folder': folder[:-1], '# records': num_rec, 
                     'eps (per record)': EPS, 'avg func L2': stats_avg[0], 
                     'avg priv err': stats_avg[1], 'avg priv time': stats_avg[2], 
                     'avg smooth err': stats_avg[3], 'avg smooth time': stats_avg[4], 
                     'avg total runtime': stats_avg[2]+stats_avg[4]}
    os.makedirs("results/tabs/", exist_ok = True)
    df.to_csv(f"results/tabs/ECG_bl_{EPS}_{SAMPLE_RATE}_{WINDOW_SCALE}.csv")
    print("-"*120)
    print("Baseline:", end = " ")
    print(f"sample size = {SAMPLE_RATE} x num datapoints;", end = " ")
    print(f"window size = {WINDOW_SCALE} x sample size.")
    print(f'''{f"avg ||f-f_bl|| = {stats_total[1].sum()/total_rec:>11.5f};":>33}''', end = "")
    print(f'''{f"avg baseline runtime = {stats_total[2].sum()/total_rec*1000:>9.5f} ms.":>87}''')
    print(f'''{f"avg ||f-f_bl_sm|| = {stats_total[3].sum()/total_rec:>11.5f};":>33}''', end = "")
    print(f'''{f"avg baseline smooth time = {stats_total[4].sum()/total_rec*1000:>9.5f} ms.":>87}''')
    print(f'''{f"avg baseline total runtime = {(stats_total[2]+stats_total[4])/total_rec*1000:>9.5f} ms.":>120}''')

if sys.argv[1] in ["t", "T", "taxi", "Taxi"]:
    for EPS in [0.001, 0.003, 0.01, 0.03, 0.1]:
        getTaxiRes(sys.argv[2], EPS)
        for SAMPLE_RATE in [0.1, 0.2]:
            for WINDOW_SCALE in [0.05, 0.1]:
                getTaxiBLRes(sys.argv[2], EPS, SAMPLE_RATE, WINDOW_SCALE)
        print("="*120+"\n")

elif sys.argv[1] in ["e", "E", "ecg", "ECG"]:
    for EPS in [0.25, 0.5, 1.0, 2.0, 4.0]:
        getECGRes(sys.argv[2], EPS, 20)
        for SAMPLE_RATE in [0.1, 0.2]:
            for WINDOW_SCALE in [0.05, 0.1]:
                getECGBLRes(sys.argv[2], EPS, SAMPLE_RATE, WINDOW_SCALE)
        print("="*120+"\n")