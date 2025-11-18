import sys
import wfdb
import shutil
from PrivPwcApprox import *

EPS = 1
BATCH_SIZE = 10
TIME_SCALE = 80
VAL_SCALE = 1000
repeat = 20
unbounded = None
parallel = None
interactive = True

if len(sys.argv) > 1:
    EPS = float(sys.argv[1])
    BATCH_SIZE = int(sys.argv[2])
    unbounded = sys.argv[3]
    parallel = sys.argv[4]
    interactive = False
    # interactive = True

records = []
timer = time.time()
# for i in range(1, 21838):
for i in range (1, 20):  # sample: run only the first 20 records
    folder = "{:05d}".format(i//1000*1000)
    file = "{:05d}_lr".format(i)
    path = "ptb-xl/records100/"+folder+"/"+file
    try:
        record = wfdb.rdrecord(path)
        records.append((folder, file, record))
    except:
        pass
print("="*50)
print(f"{len(records)} records loaded in {time.time()-timer:.2f} sec.")
print(f"Each {records[0][2].p_signal.shape[0]} datapoints, sampled at frequency {records[0][2].fs} Hz.")
print("="*50)

while unbounded not in ["Y", "y", "N", "n"]:
    unbounded = input("Allow unbounded basis function? (Y/n)\t")
while parallel not in ["Y", "y", "N", "n"]:
    parallel = input("Enable multiprocessing? (Y/n)\t")
unbounded = unbounded in ["Y", "y"]
parallel = parallel in ["Y", "y"]

for folder, file, record in records:
    timer = time.time()
    n = record.p_signal.shape[0]
    T = n//record.fs*TIME_SCALE
    t = np.linspace(1/record.fs*TIME_SCALE, T, n)
    val = record.p_signal[:, 1]*VAL_SCALE
    func = time_series_func(t, val)

    # wfdb.plot_wfdb(record = record, title = "ECG Recording")
    # plt.show()

    if unbounded:
        INTLIM_PER_PIECE = INTLIM
        solver = PrivatePiecewiseApprox((t[0], T), [t[0], T], 'Sinc-unbounded', T, parallel = parallel)
        # this is slow because for each basis, we need to compute its integral with func on [t[0], T]
        # the overhead of integartion (espcially on func that is not smooth) is very large
        # also note that splitting into intervals and use unbounded sinc (as below) will give wrong answers
        # because this will only integrate func and basis on one interval, but basis should be unbounded
        ## breakpoints = np.linspace(t[0], T, n//BATCH_SIZE)
        ## solver = PrivatePiecewiseApprox((t[0], T), breakpoints, 'Sinc-unbounded', BATCH_SIZE*TIME_SCALE//record.fs)
    else:
        breakpoints = np.linspace(t[0], T, n//BATCH_SIZE)
        solver = PrivatePiecewiseApprox((t[0], T), breakpoints, 'Sinc', BATCH_SIZE*TIME_SCALE//record.fs, parallel = parallel)
    if interactive:
        print(f"Inner product matrix preprocessed in {time.time()-timer:.2f} sec.")
    solver.fit(func, parallel = parallel)

    if interactive:
        solver.privatize(EPS)
        approx = solver.createApprox()
        priv = solver.createPriv()
        plt.figure(figsize = (20, 5))
        plt.plot(t/TIME_SCALE, val/VAL_SCALE, color = 'black')
        dense_t = np.linspace(0, T, 10*n+1)
        plt.plot(dense_t/TIME_SCALE, approx(dense_t)/VAL_SCALE, color = 'red')
        plt.plot(dense_t/TIME_SCALE, priv(dense_t)/VAL_SCALE, color = 'blue')
        print(f"Total time elapsed: {time.time()-timer:.2f} sec.")
        print(f"||f-f_approx|| = {solver.eval('Approx'):.5f};", end = " ")
        print(f"||f_approx-f_priv|| = {solver.evalPrivLoss():.5f};", end = " ")
        print(f"||f-f_priv|| = {solver.eval('Priv'):.5f}.")
        plt.show()
        break
    else:
        dir = f"results/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//record.fs}x{n//BATCH_SIZE}/"
        os.makedirs(dir, exist_ok = True)
        res_file = open(dir+f"{file}.txt", 'w')
        res_file.write(f"{np.sqrt(solver.funcSqrInt)}\n")
        res_file.write(f"{solver.eval('Approx')}\n")
        for i in range(repeat):
            solver.privatize(EPS)
            err_priv = solver.eval('Priv')
            priv_loss = solver.evalPrivLoss()
            res_file.write(f"{priv_loss} {err_priv}\n")
        res_file.close()
        print(f"Record {file} done. Executed in {time.time()-timer:.2f} sec.")

if not interactive:
    dir = f"results/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//record.fs}x{n//BATCH_SIZE}/"
    shutil.copyfile("info.log", dir+"info.log")