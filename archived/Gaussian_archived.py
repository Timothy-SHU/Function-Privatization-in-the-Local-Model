from PrivPwcApprox import *
from AdaptApprox import *

solver = PrivatePiecewiseApprox((0, 100), [0, 100], degree = 3)
func = lambda x: 100*np.exp(-(x-20)**2/(2*8*8))+200*np.exp(-(x-80)**2/(2*5*5))
approx, priv = adaptive_poly_approx(func = func, interval = (0, 100), basis = 'Polynomial', degree = 3)
plt.figure(figsize = (10, 6))
plot_1D_1D(func, approx, priv, solver)
plt.tight_layout()
plt.show()

solver = PrivatePiecewiseApprox((0, 100), [0, 25, 50, 75, 100], degree = 3)
plt.figure(figsize = (16, 12))
plt.subplot(2, 2, 1)
func = lambda x: 1*np.exp(-(x-20)**2/(2*8*8))+2*np.exp(-(x-80)**2/(2*5*5))
approx, priv = solver.solve(func, 0.5, method = 'Laplace')
plot_1D_1D(func, approx, priv, solver)
plt.subplot(2, 2, 2)
func = lambda x: 10*np.exp(-(x-20)**2/(2*8*8))+20*np.exp(-(x-80)**2/(2*5*5))
approx, priv = solver.solve(func, 0.5, method = 'Laplace')
plot_1D_1D(func, approx, priv, solver)
plt.subplot(2, 2, 3)
func = lambda x: 100*np.exp(-(x-20)**2/(2*8*8))+200*np.exp(-(x-80)**2/(2*5*5))
approx, priv = solver.solve(func, 0.5, method = 'Laplace')
plot_1D_1D(func, approx, priv, solver)
plt.subplot(2, 2, 4)
func = lambda x: 1000*np.exp(-(x-20)**2/(2*8*8))+2000*np.exp(-(x-80)**2/(2*5*5))
approx, priv = solver.solve(func, 0.5, method = 'Laplace')
plot_1D_1D(func, approx, priv, solver)
plt.tight_layout()
plt.show()

print("="*100)
print("Impact of scaling function value (1x, 10x, 100x):")
solver = PrivatePiecewiseApprox((0, 100), np.linspace(0, 100, 4), degree = 3)
for scale in [1, 10, 100]:
    func = lambda x: scale*(10*np.exp(-(x-20)**2/(2*8*8))+20*np.exp(-(x-80)**2/(2*5*5)))
    err_ls_sum = 0; err_priv_sum = 0; err_total_sum = 0
    for i in range(10):
        approx, priv = solver.solve(func, method = 'Laplace')
        err_ls_sum += l2_dist(func, approx, 0, 100)
        err_priv_sum += l2_dist(approx, priv, 0, 100)
        err_total_sum += l2_dist(func, priv, 0, 100)
    print(f"||f-f_approx|| = {err_ls_sum/10:.5f};", end = ' ')
    print(f"||f_approx-f_priv|| = {err_priv_sum/10:.5f};", end = ' ')
    print(f"||f-f_priv|| = {err_total_sum/10:.5f}.")

print("="*100)
print("Impact of scaling independent variable (1x, 10x, 100x):")
for scale in [1, 10, 100]:
    func = lambda x: 10*np.exp(-(x-0.2*scale)**2/(2*0.08*scale*0.08*scale))+20*np.exp(-(x-0.8*scale)**2/(2*0.05*scale*0.05*scale))
    solver = PrivatePiecewiseApprox((0, 1*scale), np.linspace(0, 1*scale, 4), degree = 3)
    err_ls_sum = 0; err_priv_sum = 0; err_total_sum = 0
    for i in range(10):
        approx, priv = solver.solve(func, method = 'Laplace')
        err_ls_sum += l2_dist(func, approx, 0, 1*scale)
        err_priv_sum += l2_dist(approx, priv, 0, 1*scale)
        err_total_sum += l2_dist(func, priv, 0, 1*scale)
    print(f"||f-f_approx|| = {err_ls_sum/10:.5f};", end = ' ')
    print(f"||f_approx-f_priv|| = {err_priv_sum/10:.5f};", end = ' ')
    print(f"||f-f_priv|| = {err_total_sum/10:.5f}.")