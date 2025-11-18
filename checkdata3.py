import numpy as np
from utils import Extract_Cab_Data_All
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.path import Path
from time import time

from PrivPwcApprox import *

SHIFT_X = 1.365e4#1.35e4#0.0#1.35e7
SHIFT_Y = -4.5e3#0.0#-4.5e6


def get_coeff_2d(tup0, tup1):
    t0 = tup0[0]
    t1 = tup1[0]
    dt = t1-t0
    a11 = (tup1[1]-tup0[1])/dt
    a21 =  (tup0[1]*t1-tup1[1]*t0)/dt
    a12 = (tup1[2]-tup0[2])/dt
    a22 = (tup0[2]*t1-tup1[2]*t0)/dt
    
    return [a11, a12, a21, a22]

def linfunc_prod_integ(traj, Is):
    func_t = lambda T0, T1, a1, a2: a1*(T1**3-T0**3)/3.0+a2*(T1**2-T0**2)/2
    func_1 = lambda T0, T1, a1, a2: a1*(T1**2-T0**2)/2+a2*(T1-T0)
    prod_t_x = []
    prod_1_x = []
    prod_t_y = []
    prod_1_y = []
    curr_j = 1
    Ts = traj[:,0]
    for I in Is:
        I0 = I[0]
        I1 = I[1]
        while (Ts[curr_j]<I0):
            curr_j = curr_j+1
        Ilow = max(Ts[curr_j-1],I0)
        Ihigh = min(Ts[curr_j],I1)
        [a11, a12, a21, a22] = get_coeff_2d(traj[curr_j-1],traj[curr_j])
        int_t_x = func_t(Ilow, Ihigh, a11, a21)
        prod_t_x.append(int_t_x)
        int_t_y = func_t(Ilow, Ihigh, a12, a22)
        prod_t_y.append(int_t_y)
        int_1_x = func_1(Ilow, Ihigh, a11, a21)
        prod_1_x.append(int_1_x)
        int_1_y = func_1(Ilow, Ihigh, a12, a22)
        prod_1_y.append(int_1_y)
    
    return prod_t_x, prod_t_y, prod_1_x, prod_1_y
        

def linfunc_norm(traj):
    n = len(traj)
    s = 0.0
    for j in range(1,n):
        tup0 = traj[j-1]
        tup1 = traj[j]
        t0 = tup0[0]
        t1 = tup1[0]
        dt = t1-t0
        if dt > 0:
            # a11 = (tup1[1]-tup0[1])/dt
            # a21 =  (tup0[1]*t1-tup1[1]*t0)/dt
            # a12 = (tup1[2]-tup0[2])/dt
            # a22 = (tup0[2]*t1-tup1[2]*t0)/dt
            [a11, a12, a21, a22] = get_coeff_2d(tup0,tup1)
            inc_x = a11*a11*(t1**3-t0**3)/3.0+(a11*a21)*(t1**2-t0**2)+a21*a21*dt
            inc_y = a12*a12*(t1**3-t0**3)/3.0+(a12*a22)*(t1**2-t0**2)+a22*a22*dt
            # inc = (a11*a11+a12*a12)*(t1**3-t0**3)/3.0+(a21*a11+a22*a12)*(t1**2-t0**2)+(a21*a21+a22*a22)*dt
            s = s + inc_x+inc_y
    return np.sqrt(s)

def norm_traj(traj,dt_start,dt_end,T):
    dt_T = (dt_end-dt_start).total_seconds()
    traj_out = []
    for tup in traj:
        t = (tup[0]-dt_start).total_seconds()/dt_T
        if (t>=0 and t <=1):
            traj_out.append((t*T,tup[1]+SHIFT_X,tup[2]+SHIFT_Y))
        
    return np.array(traj_out)


FOLDER_IN = './cabshort/'#'./cabspottingdata/'
T = 1
dict_traj = Extract_Cab_Data_All(folder_in=FOLDER_IN,len_min=100,len_max=50000000,R=6371)
dt_start = datetime(2008,6,1,6,0,0)
dt_end = datetime(2008,6,2,6,0,0)
names = list(dict_traj.keys())
traj_t = norm_traj(dict_traj[names[0]],dt_start=dt_start,dt_end=dt_end,T=T)



tick = time.time()
tt = traj_t[:,0]
func_x = time_series_func(tt, traj_t[:,1])
integrand_x = lambda t: func_x(t)**2
integral_x, _ = quad(integrand_x,tt[0], tt[-1], limit = INTLIM)
func_y = time_series_func(tt, traj_t[:,2])
integrand_y = lambda t: func_y(t)**2
integral_y, _ = quad(integrand_y, tt[0], tt[-1], limit = INTLIM)
print('Finished. Time elapsed: ',time.time() - tick)
print(f"traj norm :\t{np.sqrt(integral_x+integral_y):.5f}.")
tick = time.time()
ss = linfunc_norm(traj_t)
print('Finished. Time elapsed: ',time.time() - tick)
print("traj norm (2)", ss)


# traj_verts = [[tup[1],tup[2]] for tup in traj_t]
# traj_path = Path(traj_verts)
# xs, ys = zip(*traj_verts)
# plt.plot(xs, ys, '.--', lw=2, color='black', ms=10)
# # plt.scatter(traj_t[:,1],traj_t[:,2])
# plt.show()
# print(traj_t)