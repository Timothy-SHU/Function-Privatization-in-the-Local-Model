import wfdb
from PrivPwcApprox import *
from AdaptApprox import *

record = wfdb.rdrecord("ptb-xl/records100/00000/00001_lr")
# wfdb.plot_wfdb(record = record, title = "ECG Recording")
# plt.show()

BATCH_SIZE = 50
TIME_SCALE = 80
VAL_SCALE = 1000

record.p_signal = record.p_signal[:500]

timer = time.time()
iter_timer = time.time()
approx_val = []; priv_val = []
for i in range(record.p_signal.shape[0]//BATCH_SIZE):
    T = BATCH_SIZE/record.fs*TIME_SCALE
    t = np.linspace(1/record.fs, T, BATCH_SIZE)
    val = record.p_signal[i*BATCH_SIZE:(i+1)*BATCH_SIZE, 0]*VAL_SCALE
    func = time_series_func(t, val)
    solver = PrivatePiecewiseApprox((0, T), [0, T], 'Sinc', BATCH_SIZE*TIME_SCALE//record.fs)
    approx, priv = solver.solve(func, eps = 1, method = 'Laplace')
    # print(l2_dist(func, approx, 0, T), l2_dist(func, priv, 0, T))
    dense_t = np.linspace(1/record.fs, T, INTLIM)
    approx_val += (approx(dense_t)/VAL_SCALE).tolist()
    priv_val += (priv(dense_t)/VAL_SCALE).tolist()
    print(f"Batch #{i+1} done. Executed in {time.time()-iter_timer:.5f} sec.")
    iter_timer = time.time()
print(f"Total time elapsed: {time.time()-timer:.5f} sec.")
plt.figure(figsize = (20, 5))
T = record.p_signal.shape[0]/record.fs
t = np.linspace(1/record.fs, T, record.p_signal.shape[0])
plt.plot(t, record.p_signal[:, 0], color = 'black')
dense_t = np.linspace(1/record.fs, T, len(approx_val))
plt.plot(dense_t, approx_val, color = 'red')
plt.plot(dense_t, priv_val, color = 'blue')
plt.show()
