import os
import time
import numpy as np
import pandas as pd
from math import log
from json import loads
from scipy.stats import laplace, gamma
from scipy.linalg import inv, pinv, sqrtm, solve
from scipy.integrate import quad, IntegrationWarning

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

import logging
import warnings
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.DEBUG)
warnings.simplefilter("ignore", category = IntegrationWarning)      # ignore integration precision warnings

INTLIM = 1000
INTLIM_PER_PIECE = 100

class PrivatePiecewiseApprox:
    def __init__(self, interval, breakpoints, basis_type = 'Polynomial', degree = 1):
        self.degree = degree
        self.l, self.r = np.float64(interval)
        self.breakpoints = breakpoints
        self.basis_type = basis_type
        self.isOrthonormal = False

        self.params = []
        for i in range(len(breakpoints)-1):
            self.params.append([])
            for k in range(degree+1):
                self.params[-1].append({'l': breakpoints[i], 'r': breakpoints[i+1], 'k': k})
                if basis_type == 'Fourier' and k > 0:
                    self.params[-1].append({'l': breakpoints[i], 'r': breakpoints[i+1], 'k': -k})

        if basis_type == 'Polynomial':
            self.basis = lambda i, j, x: np.where(
                (x >= self.params[i][j]['l']) & ((x < self.params[i][j]['r']) | ((self.params[i][j]['r'] == self.r) & (x == self.r))), 
                np.sqrt((2*self.params[i][j]['k']+1)/2) * 
                1/np.sqrt(((self.params[i][j]['r']-self.params[i][j]['l'])/2)**(2*self.params[i][j]['k']+1)) * 
                ((x-(self.params[i][j]['l']+self.params[i][j]['r'])/2)**self.params[i][j]['k']), 
                0)
        elif basis_type == 'Fourier':
            self.basis = lambda i, j, x: np.where(
                (x >= self.params[i][j]['l']) & ((x < self.params[i][j]['r']) | ((self.params[i][j]['r'] == self.r) & (x == self.r))), 
                (self.params[i][j]['k'] == 0) * 1 +
                (self.params[i][j]['k'] > 0) * np.sin(2*np.pi*self.params[i][j]['k']*(x-(self.params[i][j]['l']+self.params[i][j]['r'])/2)) + 
                (self.params[i][j]['k'] < 0) * np.cos(-2*np.pi*self.params[i][j]['k']*(x-(self.params[i][j]['l']+self.params[i][j]['r'])/2)), 
                0)
        elif basis_type == 'Sinc' or basis_type == 'Sinc-unbounded':
            self.basis = lambda i, j, x: np.where(
                (x >= self.params[i][j]['l']) & ((x < self.params[i][j]['r']) | ((self.params[i][j]['r'] == self.r) & (x == self.r))), 
                np.sinc(x-self.params[i][j]['l']-self.params[i][j]['k']), 
                0)
        else:
            logging.error(f"ERR: no such basis '{basis_type}'.")

        self.m = len(breakpoints)-1     # m pieces in total
        self.d = len(self.params[0])    # each piece has d basis functions
        if basis_type == 'Polynomial' and degree == 1:
            self.isOrthonormal = True
        elif basis_type == 'Sinc-unbounded':
            self.isOrthonormal = True
        else:
            self.G = np.zeros((self.m, self.d, self.d))
            self.invG = np.zeros((self.m, self.d, self.d))
            for i in range(len(self.breakpoints)-1):
                for j1 in range(self.d):
                    for j2 in range(self.d):
                        l = self.params[i][j1]['l']
                        r = self.params[i][j1]['r']
                        if l != self.params[i][j2]['l'] or r != self.params[i][j2]['r']:
                            logging.error("ERR: basis functions on the same piece have different endpoints!")
                        integrand = lambda x: self.basis(i, j1, x)*self.basis(i, j2, x)
                        self.G[i, j1, j2], _ = quad(integrand, l, r, limit = INTLIM_PER_PIECE)
                self.invG[i] = pinv(self.G[i])
                # np.set_printoptions(precision = 4, linewidth = 100, suppress = True)
                # logging.info("G = \n%s", self.G[i])
                # logging.info("inv(G) = \n%s", self.invG[i])
                if np.any(np.linalg.eigvals(self.G[i]) < -1e-8):
                    logging.error("ERR: the pairwise inner product matrix is not SPSD!")
        
    def solve(self, func, eps = 0.5, method = 'Laplace'):
        b = np.zeros((self.m, self.d))
        for i in range(self.m):
            for j in range(self.d):
                integrand = lambda x: func(x)*self.basis(i, j, x)
                b[i, j], _ = quad(integrand, self.params[i][j]['l'], self.params[i][j]['r'], limit = INTLIM_PER_PIECE)
        if self.isOrthonormal:
            coeff = b
        else:
            coeff = np.zeros((self.m, self.d))
            for i in range(self.m):
                coeff[i] = self.invG[i]@b[i]
        # logging.info("b = %s", b)
        # logging.info("c = %s", coeff)
        
        def approx(x):
            ret = 0
            for i in range(self.m):
                for j in range(self.d):
                    ret += coeff[i, j]*self.basis(i, j, x)
            return ret

        if method == None:
            noise = np.zeros((self.m, self.d))
        elif method == 'Laplace':
            noise = np.random.randn(self.m*self.d)
            noise = noise/np.linalg.norm(noise)
            noise = noise.reshape(self.m, self.d)
            noise *= gamma.rvs(a = self.m*self.d, scale = 1/eps)
            if not self.isOrthonormal:
                for i in range(self.m):
                    noise[i] = (sqrtm(self.invG[i])@noise[i]).real
        elif method == 'Normal':
            noise = np.random.randn(self.m*self.d)
            noise = noise.reshape(self.m, self.d)
            if not self.isOrthonormal:
                noise[i] = (sqrtm(self.invG[i])@noise[i]).real
            noise /= np.sqrt(2*eps)
        else:
            logging.error(f"ERR: no such method '{method}'.")
        # logging.info("noise = %s", noise)

        def priv_approx(x):
            ret = 0
            for i in range(self.m):
                for j in range(self.d):
                    ret += (coeff[i, j]+noise[i, j])*self.basis(i, j, x)
            return ret

        return approx, priv_approx
    
    def eval(self, func, approx, priv_approx):
        integrand = lambda x: (func(x)-approx(x))**2
        err_ls, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        err_ls = np.sqrt(err_ls)
        integrand = lambda x: (func(x)-priv_approx(x))**2
        err_priv, _ = quad(integrand, self.l, self.r, limit = INTLIM)
        err_priv = np.sqrt(err_priv)
        return err_ls, err_priv

def l2_dist(f, g, l, r):
    integrand = lambda x: (f(x)-g(x))**2
    dist, _ = quad(integrand, l, r, limit = INTLIM)
    dist = np.sqrt(dist)
    return dist

def time_series_func(t, val):
    def func(x):
        return np.interp(x, t, val)
    return func

def plot_1D_1D(func, approx, priv, solver):
    x_plot = np.linspace(solver.l, solver.r, INTLIM)
    y_true = func(x_plot)
    y_approx = approx(x_plot)
    y_priv = priv(x_plot)
    err_ls = l2_dist(func, approx, solver.l, solver.r)
    err_priv = l2_dist(func, priv, solver.l, solver.r)
    plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'Func')
    plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Approx')
    plt.plot(x_plot, y_priv, 'b--', linewidth = 1.5, label = f'Priv Approx')
    plt.title(f"{solver.basis_type} basis, degree = {solver.degree}, error: {err_priv:.5f} ({err_ls:.5f})")
    plt.legend(loc = 'upper left')
    plt.grid(True)
