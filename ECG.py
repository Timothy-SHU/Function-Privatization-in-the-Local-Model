import wfdb
from PrivPwcApprox import *

BATCH_SIZE = 10
TIME_SCALE = 80
VAL_SCALE = 1000

timer = time.time()
record = wfdb.rdrecord("ptb-xl/records100/00000/00001_lr")
print("="*50)
print(f"Record loaded in {time.time()-timer:.5f} sec.")
print(f"{record.p_signal.shape[0]} datapoints, sampled at frequency {record.fs} Hz.")
# wfdb.plot_wfdb(record = record, title = "ECG Recording")
# plt.show()
print("="*50)

# record.p_signal = record.p_signal[600:]

n = record.p_signal.shape[0]
T = n//record.fs*TIME_SCALE
t = np.linspace(1/record.fs*TIME_SCALE, T, n)
val = record.p_signal[:, 1]*VAL_SCALE
func = time_series_func(t, val)

mode = input("Allow unbounded basis function? (Y/n)\t")
while mode not in ["Y", "y", "N", "n"]:
    mode = input("Allow unbounded basis function? (Y/n)\t")

if mode == "Y" or mode == "y":
    timer = time.time()
    INTLIM_PER_PIECE = INTLIM
    solver = PrivatePiecewiseApprox((t[0], T), [t[0], T], 'Sinc-unbounded', T)
    # this is slow because for each basis, we need to compute its integral with func on [t[0], T]
    # the overhead of integartion (espcially on func that is not smooth) is very large
    # also note that splitting into intervals and use unbounded sinc (as below) will give wrong answers
    # because this will only integrate func and basis on one interval, but basis should be unbounded
    # breakpoints = np.linspace(t[0], T, n//BATCH_SIZE)
    # solver = PrivatePiecewiseApprox((t[0], T), breakpoints, 'Sinc-unbounded', BATCH_SIZE*TIME_SCALE//record.fs)
else:
    timer = time.time()
    breakpoints = np.linspace(t[0], T, n//BATCH_SIZE)
    solver = PrivatePiecewiseApprox((t[0], T), breakpoints, 'Sinc', BATCH_SIZE*TIME_SCALE//record.fs)
print(f"Inner product matrix preprocessed in {time.time()-timer:.5f} sec.")
solver.solve(func, eps = 1, method = 'Laplace')
approx = solver.createApprox()
priv = solver.createPriv()
plt.figure(figsize = (20, 5))
plt.plot(t/TIME_SCALE, val/VAL_SCALE, color = 'black')
dense_t = np.linspace(0, T, 10*n+1)
plt.plot(dense_t/TIME_SCALE, approx(dense_t)/VAL_SCALE, color = 'red')
plt.plot(dense_t/TIME_SCALE, priv(dense_t)/VAL_SCALE, color = 'blue')
print(f"Total time elapsed: {time.time()-timer:.5f} sec.")
print(f"||f-f_approx|| = {solver.eval('Approx'):.5f};", end = " ")
print(f"||f_approx-f_priv|| = {solver.evalPrivLoss():.5f};", end = " ")
print(f"||f-f_priv|| = {solver.eval('Priv'):.5f}.")
plt.show()