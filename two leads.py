# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 00:04:43 2021

@author: Nick
"""


import numpy as np
import scipy as s
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
import scipy.linalg as sl
from scipy.integrate import quad
gamma:complex = 1+0j
gamma2=1
gamma3=1
gamma4=gamma2+gamma3
beta=100
epsilon = 2
rho=0
band_D: complex =100+0j
mu1=10
mu2=0
plt.rc('text', usetex=True)
#@jit(nopython=True, parallel=True)
def solve_dyson(arr1,arr2,step,size):
       
       G=np.linalg.inv(arr1-step*step*arr2)
       return G

def latex_float(f):
    float_str = "{0:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str
Dt=1e-3
x = np.arange(0,2,Dt)

xx,yy= np.meshgrid(x,x)
N=x.shape[0]

h1=np.exp(1j*epsilon*Dt)
h2=np.exp(-1j*epsilon*Dt)
fig,ax=plt.subplots(1,1)
fig2,ax2=plt.subplots(1,1)

def period(a):
    u1=np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    u2=np.where(u1)[0]
    return x[u2[0]]-x[u2[1]]

def f1(xy):
    mask = xy != 0
    limit = (mu1+band_D)/ (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (np.exp(-mu1*1j*xy) - np.exp(1j * band_D * xy)), xy, where=mask), limit)

def f2(xy):
    mask = xy != 0
    limit = (mu1-band_D) / (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (np.exp(-mu1*1j*xy)- np.exp(-1j * band_D * xy)), xy, where=mask), limit)
def f3(xy):
    mask = xy != 0
    limit = (mu2+band_D)/ (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (np.exp(-mu2*1j*xy) - np.exp(1j * band_D * xy)), xy, where=mask), limit)

def f4(xy):
    mask = xy != 0
    limit = (mu2-band_D) / (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (np.exp(-mu2*1j*xy) - np.exp(-1j * band_D * xy)), xy, where=mask), limit)
def g_bare_setup(h1,h2,rho):
    ginv11=-np.identity(N,dtype=complex)
    ginv22=-np.identity(N,dtype=complex)
    np.fill_diagonal(ginv11[1:,:],h1)
    np.fill_diagonal(ginv22[:,1:],h2)
    ginv12=np.zeros((N,N))
    ginv21=np.zeros((N,N))
    ginv12[0,0]=-rho
    ginv21[N-1,N-1]=1
    ginv=-1j*np.block([
        [ginv11,ginv12],
        [ginv21,ginv22]])
    return ginv

sigma=np.zeros((2*N,2*N), dtype=complex)
sigma[0:N,N:2*N]=-1j*gamma2*f1(xx - yy) -1j*gamma3*f3(xx-yy)
sigma[N:2*N,0:N]=-1j*gamma2*f2(xx - yy) - 1j*gamma3*f4(xx-yy)
sigma[N:2*N,N:2*N]=-(np.heaviside(xx-yy,0.1)*sigma[0:N,N:2*N]+np.heaviside(yy-xx,0.1)*sigma[N:2*N,0:N])

sigma[0:N,0:N]=-(np.heaviside(xx-yy,0.1)*sigma[N:2*N,0:N]+np.heaviside(yy-xx,0.1)*sigma[0:N,N:2*N])

ginv=g_bare_setup(h1,h2,rho)
G=solve_dyson(ginv,sigma,Dt,N)
nvals=np.zeros_like(x)

integrand_p1 = lambda x, t:  s.special.expit(-beta*(x-mu1))
integrand_p2 = lambda x,t : (1+np.exp(-2*gamma4*t)-2*np.exp(-gamma4*t)*np.cos((x-epsilon)*t))/(gamma4**2 +(x-epsilon)**2)
n1 = lambda t:  gamma2/((math.pi))*(quad(lambda x: (integrand_p1(x,t)*integrand_p2(x,t)), -100, mu1, limit=300,limlst=300))[0]
integrand_p3 = lambda x, t:  s.special.expit(-beta*(x-mu2))
integrand_p4 = lambda x,t : (1+np.exp(-2*gamma4*t)-2*np.exp(-gamma4*t)*np.cos((x-epsilon)*t))/(gamma4**2 +(x-epsilon)**2)
n2 = lambda t:  gamma3/((math.pi))*(quad(lambda x: (integrand_p3(x,t)*integrand_p4(x,t)), -100, mu2, limit=300,limlst=300))[0]
nvals1=np.vectorize(n1)(x)
nvals2=np.vectorize(n2)(x)
nvals=nvals1+nvals2
n1=np.diagonal(-1j*G[0:N,N:2*N])
ax.plot(x[30:], 100*np.abs(n1[30:]-nvals[30:])/nvals[30:])
#ax.plot(x,nvals, label='analytic')
#ax.plot(x,n1, label='inversion')
#plt.plot(x,np.diagonal(-1j*G[0:N,N:2*N]))
T=period(n1)
print("The period of the oscillations is:", T)




ax.set_ylabel(r'rel. error (\%)')
ax.set_xlabel(r'$t\,(\Gamma^{-1})$')
#ax.legend()
ax.grid()
ax.set_title(r"$ \mu_1=10\Gamma, \ \mu_2=0,\ \epsilon_d=2\Gamma$")


plt.show()
fig.savefig('twoleads_errors2.pdf')