import os
import time
import numpy as np
import pandas as pd
from math import log
from json import loads
from scipy.stats import laplace, gamma
from scipy.linalg import inv, pinv, sqrtm, solve
from scipy.integrate import quad, quad_vec, IntegrationWarning

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

import logging
import warnings
logging.getLogger('matplotlib.pyplot').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(filename = 'info.log', filemode = 'w', level = logging.INFO)
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

        self.basis = []
        for i in range(len(breakpoints)-1):
            self.basis.append([])
            for k in range(degree+1):
                self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k))
                if basis_type == 'Linear-2D':
                    self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], k, 1))
                if basis_type == 'Fourier' and k > 0:
                    self.basis[-1].append(self.createBasis(breakpoints[i], breakpoints[i+1], -k))
        
        self.m = len(breakpoints)-1     # m pieces in total
        self.d = len(self.basis[0])    # each piece has d basis functions
        if basis_type == 'Polynomial' and degree <= 2:
            self.isOrthonormal = True
        elif basis_type == 'Linear-2D':
            self.isOrthonormal = True
        elif basis_type == 'Sinc-unbounded':
            self.isOrthonormal = True
        else:
            self.G = np.zeros((self.m, self.d, self.d))
            self.invG = np.zeros((self.m, self.d, self.d))
            for i in range(self.m):
                for j1 in range(self.d):
                    for j2 in range(self.d):
                        l = breakpoints[i]
                        r = breakpoints[i+1]
                        integrand = lambda x: self.basis[i][j1](x)*self.basis[i][j2](x)
                        self.G[i, j1, j2], _ = quad(integrand, l, r, limit = INTLIM_PER_PIECE)
                self.invG[i] = pinv(self.G[i])
                # np.set_printoptions(precision = 4, linewidth = 100, suppress = True)
                # logging.info("G = \n%s", self.G[i])
                # logging.info("inv(G) = \n%s", self.invG[i])
                if np.any(np.linalg.eigvals(self.G[i]) < -1e-8):
                    logging.error("ERR: the pairwise inner product matrix is not SPSD!")
    
    def createBasis(self, l, r, k, dim = 0):
        if self.basis_type == 'Polynomial':
            if k == 0:
                return lambda x: 1/np.sqrt(r-l)
            elif k == 1:
                return lambda x: np.sqrt(12/((r-l)**3))*(x-(l+r)/2)
            elif k == 2:
                return lambda x: np.sqrt(180/((r-l)**5))*((x-(l+r)/2)**2)-np.sqrt(5/(r-l))/2
            else:
                return lambda x: np.sqrt((2*k+1)/2) * 1/np.sqrt(((r-l)/2)**(2*k+1)) * ((x-(l+r)/2)**k)
        elif self.basis_type == 'Linear-2D':
            if k == 0 and dim == 0:
                return lambda x: np.array([1/np.sqrt(r-l), 0])
            elif k == 0 and dim == 1:
                return lambda x: np.array([0, 1/np.sqrt(r-l)])
            elif k == 1 and dim == 0:
                return lambda x: np.array([np.sqrt(12/((r-l)**3))*(x-(l+r)/2), 0])
            elif k == 1 and dim == 1:
                return lambda x: np.array([0, np.sqrt(12/((r-l)**3))*(x-(l+r)/2)])
            else:
                logging.error(f"ERR: linear 2D basis param 'k = {k}, dim = {dim}' not valid.")
        elif self.basis_type == 'Fourier':
            if k == 0:
                return lambda x: 1
            elif k > 0:
                return lambda x: np.sin(2*np.pi*k*(x-(l+r)/2))
            else:
                return lambda x: np.cos(-2*np.pi*k*(x-(l+r)/2))
        elif self.basis_type == 'Sinc' or self.basis_type == 'Sinc-unbounded':
            return lambda x: np.sinc(x-l-k)
        else:
            logging.error(f"ERR: no such basis '{self.basis_type}'.")

    def solve(self, func, eps = 0.5, method = 'Laplace'):
        self.func = func
        self.eps = eps
        self.method = method

        integrand = lambda x: func(x)**2
        if self.basis_type == 'Linear-2D':
            integral_vec, _ = quad_vec(integrand, self.l, self.r, limit = INTLIM)
            self.funcSqrInt = np.sum(integral_vec)
        else:
            self.funcSqrInt, _ = quad(integrand, self.l, self.r, limit = INTLIM)

        self.b = np.zeros((self.m, self.d))
        for i in range(self.m):
            for j in range(self.d):
                l = self.breakpoints[i]
                r = self.breakpoints[i+1]
                integrand = lambda x: func(x)*self.basis[i][j](x)
                if self.basis_type == 'Linear-2D':
                    integral_vec, _ = quad_vec(integrand, l, r, limit = INTLIM_PER_PIECE)
                    self.b[i, j] = np.sum(integral_vec)
                else:
                    self.b[i, j], _ = quad(integrand, l, r, limit = INTLIM_PER_PIECE)
        if self.isOrthonormal:
            self.coeff = self.b
        else:
            self.coeff = np.zeros((self.m, self.d))
            for i in range(self.m):
                self.coeff[i] = self.invG[i]@self.b[i]
        # logging.info("b = %s", self.b)
        # logging.info("c = %s", self.coeff)

        if method == None:
            self.noise = np.zeros((self.m, self.d))
        elif method == 'Laplace':
            self.noise = np.random.randn(self.m*self.d)
            self.noise = self.noise/np.linalg.norm(self.noise)
            self.noise = self.noise.reshape(self.m, self.d)
            self.noise *= gamma.rvs(a = self.m*self.d, scale = 1/eps)
            if not self.isOrthonormal:
                for i in range(self.m):
                    self.noise[i] = (sqrtm(self.invG[i])@self.noise[i]).real
        elif method == 'Normal':
            self.noise = np.random.randn(self.m*self.d)
            self.noise = self.noise.reshape(self.m, self.d)
            if not self.isOrthonormal:
                self.noise[i] = (sqrtm(self.invG[i])@self.noise[i]).real
            self.noise /= np.sqrt(2*eps)
        else:
            logging.error(f"ERR: no such method '{method}'.")
        # logging.info("noise = %s", self.noise)
    
    def createApprox(self):
        if self.func == None:
            logging.error(f"ERR: solver not fitted but called for approximation result.")
        def approx(x):
            if isinstance(x, (int, float)):
                ret = 0
                if x >= self.l and x <= self.r:
                    i = min(np.searchsorted(self.breakpoints, x, side = 'right')-1, self.m-1)
                    for j in range(self.d):
                        ret += self.coeff[i, j]*self.basis[i][j](x)
                return ret
            else:
                return np.array([approx(e) for e in x])
        return approx

    def createPriv(self):
        if self.func == None:
            logging.error(f"ERR: solver not fitted but called for privatization result.")
        def priv(x):
            if isinstance(x, (int, float)):
                ret = 0
                if x >= self.l and x <= self.r:
                    i = min(np.searchsorted(self.breakpoints, x, side = 'right')-1, self.m-1)
                    for j in range(self.d):
                        ret += (self.coeff[i, j]+self.noise[i, j])*self.basis[i][j](x)
                return ret
            else:
                return np.array([priv(e) for e in x])
        return priv

    def eval(self, type = 'Approx'):
        # Evaluate l2-dist between func and approx / priv
        # \int(func - \sum_i(coeff[i]*basis[i]))^2
        # = func^2 - 2*\sum_i(coeff[i]*func*basis[i]) 
        #   + \sum_i\sum_j(coeff[i]*coeff[j]*basis[i]*basis[j])
        # The second term is 2*coeff@b
        # The third term is coeff^T @ G @ coeff
        err = self.funcSqrInt
        if type == 'Approx':
            for i in range(self.m):
                err -= 2*self.coeff[i]@self.b[i]
                if self.isOrthonormal:
                    err += self.coeff[i]@self.coeff[i]
                    # print(self.funcSqrInt, self.coeff[i]@self.coeff[i])
                else:
                    err += self.coeff[i].T@self.G[i]@self.coeff[i]
        elif type == 'Priv':
            for i in range(self.m):
                err -= 2*(self.coeff[i]+self.noise[i])@self.b[i]
                if self.isOrthonormal:
                    err += (self.coeff[i]+self.noise[i])@(self.coeff[i]+self.noise[i])
                else:
                    err += (self.coeff[i]+self.noise[i]).T@self.G[i]@(self.coeff[i]+self.noise[i])
        else:
            logging.error(f"ERR: no such loss type '{type}'.")
        return np.sqrt(err)

    def evalPrivLoss(self):
        # Evaluate l2-dist between approx and priv
        # \int(\sum_i(noise[i]*basis[i]))^2)
        # which is just noise^T @ G @ noise
        err = 0
        if self.isOrthonormal:
            for i in range(self.m):
                err += self.noise[i]@self.noise[i]
        else:
            for i in range(self.m):
                err += self.noise[i].T@self.G[i]@self.noise[i]
        return np.sqrt(err)

    def clear(self):
        self.func = None
        self.eps = None
        self.method = None
        self.funcSqrInt = None
        self.b = None
        self.coeff = None
        self.noise = None

def time_series_func(t, val):
    def func(x):
        return np.interp(x, t, val)
    return func

def time_series_func_2D(t, val1, val2):
    def func2D(x):
        return np.array([np.interp(x, t, val1), np.interp(x, t, val2)])
    return func2D

def plot_1D_1D(solver):
    x_plot = np.linspace(solver.l, solver.r, INTLIM)
    y_true = solver.func(x_plot)
    y_approx = solver.createApprox()(x_plot)
    y_priv = solver.createPriv()(x_plot)
    err_ls = solver.eval('Approx')
    err_priv = solver.eval('Priv')
    plt.plot(x_plot, y_true, 'k-', linewidth = 2, label = 'Func')
    plt.plot(x_plot, y_approx, 'r--', linewidth = 1.5, label = 'Approx')
    plt.plot(x_plot, y_priv, 'b--', linewidth = 1.5, label = f'Priv Approx')
    plt.title(f"{solver.basis_type} basis, degree = {solver.degree}, error: {err_priv:.5f} ({err_ls:.5f})")
    plt.legend(loc = 'upper left')
    plt.grid(True)