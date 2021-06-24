# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:02:25 2021

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

Dt=0.5*1e-2
x = np.arange(0,3,Dt)

xx,yy= np.meshgrid(x,x)
gamma:complex = 1+0j
start=50
tau=100
beta=100
epsilon = np.piecewise(x,[x <.4, x>=.4], [start, lambda x: start*np.cos(tau*(x-0.4))])
#epsilon1 = lambda x: np.heaviside(2-x,0)*start*np.cos((x-0.2)*tau*np.pi/180)
#epsilon=np.vectorize(epsilon1)(x)
rho=0
band_D: complex =100+0j

#@jit(nopython=True, parallel=True)
def solve_dyson(arr1,arr2,step,size):
       G=np.linalg.inv(arr1-step*step*arr2)
       return G





N=x.shape[0]

h1=np.exp(1j*epsilon*Dt)
h2=np.exp(-1j*epsilon*Dt)
fig,ax=plt.subplots(1,1)



def f1(xy):
    mask = xy != 0
    limit = band_D / (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (1 - np.exp(1j * band_D * xy)), xy, where=mask), limit)

def f2(xy):
    mask = xy != 0
    limit = -band_D / (np.pi)
    return np.where(mask, np.divide(1j/(np.pi) * (1 - np.exp(-1j * band_D * xy)), xy, where=mask), limit)
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
sigma[0:N,N:2*N]=-1j*gamma*f1(xx - yy)
sigma[N:2*N,0:N]=-1j*gamma*f2(xx - yy) 
sigma[N:2*N,N:2*N]=-(np.heaviside(xx-yy,0.5)*sigma[0:N,N:2*N]+np.heaviside(yy-xx,0.5)*sigma[N:2*N,0:N])

sigma[0:N,0:N]=-(np.heaviside(xx-yy,0.5)*sigma[N:2*N,0:N]+np.heaviside(yy-xx,0.5)*sigma[0:N,N:2*N])

ginv=g_bare_setup(h1,h2,rho)
G=solve_dyson(ginv,sigma,Dt,N)
ax.plot(x,np.diagonal(-1j*G[0:N,N:2*N]))
ax.set_ylabel(r'n')
ax.set_xlabel(r'$t\,(\Gamma^{-1})$')
ax.grid()
plt.title(r'$\Omega=100\Gamma$, $\epsilon_0=50\Gamma$, $\Gamma=1$')
#plt.plot(x, epsilon)
fig.savefig('sinusoidnew.pdf')
plt.show()