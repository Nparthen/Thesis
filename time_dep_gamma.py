# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:54:18 2021

@author: Nick
"""


import numpy as np
import scipy as s
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.integrate import quad
gamma:complex = 1+0j
gamma2=gamma
beta=100
epsilon: complex =20+0j
rho=0
band_D: complex =100+0j
plt.rc('text', usetex=True)
#@jit(nopython=True, parallel=True)
def solve_dyson(arr1,arr2,step,size):
       iden=np.identity(2*N)
       G=np.linalg.inv(arr1-step*step*arr2)
       return G

def latex_float(f):
    float_str = "{0:.2e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str
Dt=0.5*1e-2
x = np.arange(0,3,Dt)
y = np.arange(0,3,Dt)
xx,yy= np.meshgrid(x,y)
N=x.shape[0]

h1=np.exp(1j*epsilon*Dt)
h2=np.exp(-1j*epsilon*Dt)
fig,ax=plt.subplots(1,1)
fig2,ax2=plt.subplots(1,1)


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

coeff=np.array([1,2,3,4,10,20,1e5])
#coeff=np.arange(1,10,0.1)
#lifetime=np.zeros(coeff.shape)

for i in range(len(coeff)):
    sigma=np.zeros((2*N,2*N), dtype=complex)
    sigma[0:N,N:2*N]=-1j*gamma*np.tanh((coeff[i]*xx)**(2))*np.tanh((coeff[i]*yy)**(2))*f1(xx - yy)
    sigma[N:2*N,0:N]=-1j*gamma*np.tanh((coeff[i]*xx)**(2))*np.tanh((coeff[i]*yy)**(2))*f2(xx - yy) 
    sigma[N:2*N,N:2*N]=-(np.heaviside(xx-yy,0.5)*sigma[0:N,N:2*N]+np.heaviside(yy-xx,0.5)*sigma[N:2*N,0:N])

    sigma[0:N,0:N]=-(np.heaviside(xx-yy,0.5)*sigma[N:2*N,0:N]+np.heaviside(yy-xx,0.5)*sigma[0:N,N:2*N])

    ginv=g_bare_setup(h1,h2,rho)
    G=solve_dyson(ginv,sigma,Dt,N)
    stri=latex_float(coeff[i])
    n1=np.diagonal(-1j*G[0:N,N:2*N]).real
    ax.plot(x,n1)
    #u=np.where(abs(n1 - 0.4) < Dt/2)[0]
    #lifetime[i]=x[u[0]]
    



nvals=np.zeros_like(x)

integrand_p1 = lambda x, t:  s.special.expit(-beta*x)
integrand_p2 = lambda x,t : (1+np.exp(-2*gamma2*t)-2*np.exp(-gamma2*t)*np.cos((x-epsilon)*t))/(gamma2**2 +(x-epsilon)**2)
n = lambda t:  gamma2/(math.pi)*(quad(lambda x: (integrand_p1(x,t)*integrand_p2(x,t)), -100, 0, limit=300,limlst=300))[0]
nvals=np.vectorize(n)(x)



ax.plot(x,nvals,label=r'analytic, $\theta$-turn on')
#ax2.scatter(coeff,lifetime)
#ax2.set_xscale("log")
ax.legend(ncol=2)
ax.grid()
ax.set_title(r"$\epsilon_d=20\Gamma_0$")
ax.set_ylabel(r'$n$')
ax.set_xlabel(r'$t\,(\Gamma^{-1})$')
plt.show()
fig.savefig('arrow3.pdf')