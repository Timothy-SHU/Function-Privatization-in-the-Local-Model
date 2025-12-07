import os, sys, time, wfdb
from PrivPwcApprox import *
from tqdm import tqdm

EPS = 1.0
METHOD = 'Laplace'
BATCH_SIZE = 20
TIME_SCALE = 80
VAL_SCALE = 1000
repeat = 10
unbounded = None
parallel = None
interactive = True

def genNoise(method, scale, dim = 1):
    if method == 'Laplace':
        if dim == 1:
            noise = laplace.rvs(scale = scale)
        else:
            noise = np.random.randn(dim)
            noise = noise/np.linalg.norm(noise)
            noise *= gamma.rvs(a = dim, scale = scale)
    elif method == 'Gaussian':
        if dim == 1: noise = norm.rvs(scale = scale)
        else: noise = scale*np.random.randn(dim)
    return noise

if len(sys.argv) > 1:
    METHOD = str(sys.argv[1])
    EPS = float(sys.argv[2])
    BATCH_SIZE = int(sys.argv[3])
    unbounded = (BATCH_SIZE == -1)
    parallel = False
    interactive = False
    # interactive = True

records = []
timer = time.time()
# for i in range(1, 21838):
# for i in range(1, 1000):  # run the first folder
for i in range (513):   # run the first 500 records
# for i in range (1, 21):   # sample: run only the first 20 records
    folder = "{:05d}".format(i//1000*1000)
    file = "{:05d}_lr".format(i)
    path = "ptb-xl/records100/"+folder+"/"+file
    try:
        record = wfdb.rdrecord(path)
        records.append((folder, file, record))
    except:
        pass
print("="*80)
print(f"{len(records)} records loaded in {time.time()-timer:.2f} sec.")
print(f"Each {records[0][2].p_signal.shape[0]} datapoints, sampled at frequency {records[0][2].fs} Hz.")
print("="*80)

if len(sys.argv) == 1:
    str = input("Apply Laplace noise? [ Y(GP) / n(CGP) ]:\t")
    if str in ["N", "n"]: METHOD = 'Gaussian'
    str = input("Privacy budget (in eps) per record [default 1.0]:\t")
    if str != "": EPS = float(str)
    str = input("Batch size (default 20, input -1 for unbounded basis):\t")
    if str != "": BATCH_SIZE = int(str)
    unbounded = (BATCH_SIZE == -1)
    parallel = input("Enable multiprocessing? (y/N)\t")
    parallel = parallel in ["Y", "y"]

timer = time.time()
for folder, file, record in tqdm(records, position = 0, leave = True):
    iter_timer = time.time()
    n = record.p_signal.shape[0]
    T = n//record.fs*TIME_SCALE
    t = np.linspace(1/record.fs*TIME_SCALE, T, n)
    val = record.p_signal[:, 1]*VAL_SCALE
    func = time_series_func(t, val)
    eps = EPS
    # if METHOD == 'Gaussian': eps = eps*eps/2
    if METHOD == 'Gaussian': eps = eps*eps/125

    # wfdb.plot_wfdb(record = record, title = "ECG Recording")
    # plt.show()

    if unbounded:
        INTLIM_PER_PIECE = INTLIM
        solver = PrivatePiecewiseApprox((t[0], T), [t[0], T], 'Sinc-unbounded', int(T-t[0]), parallel = parallel)
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
        print(f"Inner product matrix preprocessed in {time.time()-iter_timer:.2f} sec.")
    solver.fit(func, (t, val), parallel = parallel)
    approx_time = time.time()-iter_timer
    err_ls = solver.eval('Approx')

    if interactive:
        solver.privatize(eps, METHOD)
        approx = solver.createApprox()
        priv = solver.createPriv()
        err_priv = solver.eval('Priv')
        priv_loss = solver.evalPrivLoss()
        print(f"Privatized in {time.time()-iter_timer:.2f} sec (incl preproc).")

        dense_t = np.linspace(t[0], T, 10*n+1)
        plt.subplot(4, 1, 1)
        plt.plot(t/TIME_SCALE, val, color = 'black', label = "function")
        plt.plot(dense_t/TIME_SCALE, approx(dense_t), color = 'tab:blue', 
                 alpha = 0.9, label = "approximation")
        plt.xlabel("time (s)"); plt.ylabel(r"amplitude ($\mu$V)"); plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(t/TIME_SCALE, val, color = 'black', label = "function")
        plt.plot(dense_t/TIME_SCALE, priv(dense_t), color = 'tab:orange', 
                 alpha = 0.9, label = "privatization")
        plt.xlabel("time (s)"); plt.ylabel(r"amplitude ($\mu$V)"); plt.legend()

        smooth_timer = time.time()
        solver.smooth()
        smooth = solver.createPriv()
        err_smooth = solver.eval('Priv')
        smooth_loss = solver.evalPrivLoss()
        print(f"Smoothed in {time.time()-smooth_timer:.2f} sec.")
        print(f"Total time elapsed: {time.time()-iter_timer:.2f} sec.")
        print(f"||f-f_approx|| = {err_ls:.5f};")
        print(f"||f_approx-f_priv|| = {priv_loss:.5f};", end = " ")
        print(f"||f-f_priv|| = {err_priv:.5f};")
        print(f"||f_approx-f_smoothed|| = {smooth_loss:.5f};", end = " ")
        print(f"||f-f_smoothed|| = {err_smooth:.5f}.")
        print("="*80)

        plt.subplot(4, 1, 3)
        plt.plot(t/TIME_SCALE, val, color = 'black', label = "function")
        plt.plot(dense_t/TIME_SCALE, smooth(dense_t), color = 'tab:purple', 
                 alpha = 0.9, label = "privatization (continuous)")
        plt.xlabel("time (s)"); plt.ylabel(r"amplitude ($\mu$V)"); plt.legend()

        SAMPLE = int(len(t)*0.1)
        WINDOW = max(1, int(SAMPLE*0.05/2))
        sample = np.linspace(0, n-1, SAMPLE, dtype = int)
        sample = [i in sample for i in range(len(t))]
        val_priv = val[sample]
        for i in range(SAMPLE):
            if METHOD == 'Laplace':
                val_priv[i] += genNoise(METHOD, SAMPLE/eps)
            elif METHOD == 'Gaussian':
                val_priv[i] += genNoise(METHOD, np.sqrt(SAMPLE/(2*eps)))
        val_smooth = np.zeros(SAMPLE)
        for l in range(SAMPLE):
            val_smooth[l] = np.mean(val_priv[max(0, l-WINDOW) : min(l+WINDOW+1, len(sample))])
        plt.subplot(4, 1, 4)
        plt.plot(t/TIME_SCALE, val, color = 'black', label = "function")
        plt.plot(t[sample]/TIME_SCALE, val_priv, color = 'tab:green', 
                 alpha = 0.9, label = "baseline")
        plt.plot(t[sample]/TIME_SCALE, val_smooth, color = 'tab:brown', 
                 alpha = 0.9, label = "baseline (smoothed)")
        plt.xlabel("time (s)"); plt.ylabel(r"amplitude ($\mu$V)"); plt.legend()
        plt.subplots_adjust(left = 0.03, right = 0.99, top = 0.99, bottom = 0.03, 
                            wspace = 0.1, hspace = 0.125) 
        plt.show()
        exit(0)
    else:
        dir = f"results/ECG/ECG_{EPS}_{BATCH_SIZE*TIME_SCALE//record.fs}x{n//BATCH_SIZE}/{folder}/"
        os.makedirs(dir, exist_ok = True)
        res_file = open(dir+f"{file}_{METHOD}.txt", 'w')
        res_file.write(f"{t[-1]-t[0]} {EPS}\n")
        res_file.write(f"{np.sqrt(solver.funcSqrInt)}\n")
        res_file.write(f"{err_ls} {approx_time}\n\n")
        for i in range(repeat):
            priv_timer = time.time()
            solver.privatize(eps, METHOD)
            priv_time = time.time()-priv_timer
            err_priv = solver.eval('Priv')
            priv_loss = solver.evalPrivLoss()
            res_file.write(f"{priv_loss} {err_priv} {priv_time}\n")
            smooth_timer = time.time()
            solver.smooth()
            smooth_time = time.time()-smooth_timer
            err_smooth = solver.eval('Priv')
            smooth_loss = solver.evalPrivLoss()
            res_file.write(f"{smooth_loss} {err_smooth} {smooth_time}\n")
            res_file.write("\n")
        res_file.close()
        print(f"Record {file} done. Executed in {time.time()-iter_timer:.2f} sec.", flush = True)

if not interactive:
    print(f"Total time elapsed: {time.time()-timer:.2f} sec.")