import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def get_sphlap_noise(d):
    assert(d>=2)
    noise = np.random.normal(loc=0.0,scale=1.0,size=d)
    theta = noise/np.linalg.norm(noise)
    mag = np.random.gamma(shape=d,scale=1.0)
    return mag*theta

############################################ vector basis 1 ############################################ 
# eps = 1.0

# ts = np.arange(0,2*np.pi,0.01)

# a = np.array([4.8,-2,-1])
# xs = a[0]*np.cos(ts)+a[1]*1.0+a[2]*0.0
# ys = a[0]*np.sin(ts)-a[1]*0.0+a[2]*1.0

# np.random.seed(seed=2027)
# noise_vec = get_sphlap_noise(3)#np.array([-np.sqrt(2.0),np.sqrt(2.0)])#
# noise_vec = noise_vec/np.sqrt(2*np.pi)/eps
# a_tilde = a+noise_vec
# print(a)
# print(a_tilde)
# xs_tilde = a_tilde[0]*np.cos(ts)+a_tilde[1]*1.0+a_tilde[2]*0.0
# ys_tilde = a_tilde[0]*np.sin(ts)-a_tilde[1]*0.0+a_tilde[2]*1.0

# plt.figure(figsize=(5.5, 5.5))
# plt.plot(xs,ys,linestyle='dashed',c='k',label='true')
# plt.plot(xs_tilde,ys_tilde,label='privatized')

# savename = './vecbasis0.pdf'
# plt.legend(loc='upper right',framealpha=0.4,frameon=False,fancybox=True)#, shadow=True)
# if savename != '':
#     plt.savefig(savename,bbox_inches='tight', pad_inches=0.02)
# else:
#     plt.show()
# print('test')


############################################ vector basis 2 ############################################ 
# eps = 1.0

# ts = np.arange(0,4*np.pi,0.01)

# a = np.array([2,1])
# xs = a[0]*np.cos(ts)+a[1]*np.cos(0.5*ts)
# ys = a[0]*np.sin(ts)-a[1]*np.sin(0.5*ts)

# np.random.seed(seed=2027)
# noise_vec = get_sphlap_noise(2)#np.array([-np.sqrt(2.0),np.sqrt(2.0)])#
# noise_vec = noise_vec/2.0/np.sqrt(np.pi)/eps
# a_tilde = a+noise_vec
# print(a_tilde)
# xs_tilde = a_tilde[0]*np.cos(ts)+a_tilde[1]*np.cos(0.5*ts)
# ys_tilde = a_tilde[0]*np.sin(ts)-a_tilde[1]*np.sin(0.5*ts)

# plt.figure(figsize=(5.5, 5.5))
# plt.plot(xs,ys,linestyle='dashed',c='k',label='true')
# plt.plot(xs_tilde,ys_tilde,label='privatized')

# savename = './vecbasis.pdf'
# plt.legend(framealpha=0.4,frameon=False,fancybox=True)#, shadow=True)
# if savename != '':
#     plt.savefig(savename,bbox_inches='tight', pad_inches=0.02)
# else:
#     plt.show()
# print('test')


############################################ vector coefficients ############################################ 
# eps = 1.0

# ts = np.arange(0,2*np.pi,0.01)

# a = np.array([[3.0,-1.0],[5.0,3.5]])
# xs = a[0][0]*np.cos(ts)+a[0][1]*np.sin(ts)
# ys = a[1][0]*np.cos(ts)+a[1][1]*np.sin(ts)

# np.random.seed(seed=2027)
# noise_vec = get_sphlap_noise(4)#np.array([-np.sqrt(2.0),np.sqrt(2.0)])#
# noise_vec = noise_vec/np.sqrt(np.pi)/eps
# a_tilde = np.array([a[0]+noise_vec[:2],a[1]+noise_vec[2:]])
# print(a_tilde)
# xs_tilde = a_tilde[0][0]*np.cos(ts)+a_tilde[0][1]*np.sin(ts)
# ys_tilde = a_tilde[1][0]*np.cos(ts)+a_tilde[1][1]*np.sin(ts)

# plt.figure(figsize=(5.5, 5.5))
# plt.plot(xs,ys,linestyle='dashed',c='k',label='true')
# plt.plot(xs_tilde,ys_tilde,label='privatized')

# savename = './veccoeff0.pdf'
# plt.legend(framealpha=0.4,frameon=False,fancybox=True)#, shadow=True)
# if savename != '':
#     plt.savefig(savename,bbox_inches='tight', pad_inches=0.02)
# else:
#     plt.show()
# # print('test')

############################################ vector coefficients 2 ############################################ 
eps = 1.0

ts = np.arange(0,4.0,0.01)

a = np.array([[3.0,-1.0],[5.0,3.5]])
xs = a[0][0]*ts+a[0][1]*1.0
ys = a[1][0]*ts+a[1][1]*1.0

np.random.seed(seed=2027)
noise_vec = get_sphlap_noise(4)#np.array([-np.sqrt(2.0),np.sqrt(2.0)])#
sig = np.array([[3.0/16.0,-6.0/16.0],[-6.0/16.0,1.0]])#np.array([[4.0/9.0,-2.0/3.0],[-2.0/3.0,4.0/3.0]])#np.array([[1.5,-1.5],[-1.5,2.0]])#
sig_sqrt = sqrtm(sig)
noise_vec0 = np.matmul(sig_sqrt,noise_vec[:2])/eps
noise_vec1 = np.matmul(sig_sqrt,noise_vec[2:])/eps
a_tilde = np.array([a[0]+noise_vec0,a[1]+noise_vec1])

print(a_tilde)
xs_tilde = a_tilde[0][0]*ts+a_tilde[0][1]*1.0
ys_tilde = a_tilde[1][0]*ts+a_tilde[1][1]*1.0

plt.figure(figsize=(5.5, 5.5))
plt.plot(xs,ys,linestyle='dashed',c='k',label='true')
plt.plot(xs_tilde,ys_tilde,label='privatized')

savename = './veccoeff.pdf'
plt.legend(framealpha=0.4,frameon=False,fancybox=True)#, shadow=True)
if savename != '':
    plt.savefig(savename,bbox_inches='tight', pad_inches=0.02)
else:
    plt.show()
# print('test')





