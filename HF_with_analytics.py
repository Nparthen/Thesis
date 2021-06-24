# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:02:12 2021

@author: Nick
"""

import numpy as np
import scipy as s
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.linalg as sl
from scipy.integrate import quad
import time
import warnings
from scipy.optimize import fsolve
from tempfile import TemporaryFile
outfile = TemporaryFile()
warnings.filterwarnings('ignore')
gamma:complex = 1+0j
beta=100
U=5
epsilon = 20
rho_up=0
rho_down=0
band_D: complex =100+0j
Dt=0.1*1e-2
x = np.arange(0,2,Dt)
xx,yy= np.meshgrid(x,x)
N=x.shape[0]
fig,ax=plt.subplots(1,1)
h1=np.exp(1j*epsilon*Dt)
h2=np.exp(-1j*epsilon*Dt)
fig,ax=plt.subplots(1,1)
di=np.diag_indices(N)
plt.rc('text', usetex=True)


def solve_dyson(arr1,arr2,step,size):
  
       G=np.linalg.inv(arr1-step*step*arr2)
       
       return G





def f1(xy):
    """This function evaluates the tunneling +- self-energy component for T=0 (exact ), where the Fermi function 
    is modeled as a Heaviside theta."""
    
    mask = xy != 0
    
    limit = band_D / (np.pi)
    
    return np.where(mask, np.divide(1j/(np.pi) * (1 - np.exp(1j * band_D * xy)), xy, where=mask), limit)


def f2(xy):
    """This function evaluates the tunneling -+ self-energy component for T=0 (exact ), where the Fermi function 
    is modeled as a Heaviside theta."""
    
    mask = xy != 0
    
    limit = -band_D / (np.pi)
    
    return np.where(mask, np.divide(1j/(np.pi) * (1 - np.exp(-1j * band_D * xy)), xy, where=mask), limit)


def g_bare_setup(h1,h2,rho):
    """This function sets up the bare propagator in Keldysh space in the +,- basis. Contour ordering for all Keldysh 
    blocks follows the conventional array indexing."""
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


def equations(vars):
    x, y = vars
    eq1 = x - 1/2 + 1/math.pi*(s.arctan(epsilon+U*y))
    eq2 =  y- 1/2 + 1/math.pi*(s.arctan(epsilon+U*x))
    return [eq1, eq2]

x1, y1 =  fsolve(equations, (0, 1))

print(x1, y1)
"""Setup and evaluate the tuneling self-energy, based on the Keldysh relations."""
sigma=np.zeros((2*N,2*N), dtype=complex)
sigma[0:N,N:2*N]=-1j*gamma*f1(xx - yy)
sigma[N:2*N,0:N]=-1j*gamma*f2(xx - yy) 
sigma[N:2*N,N:2*N]=-(np.heaviside(xx-yy,0.5)*sigma[0:N,N:2*N]+np.heaviside(yy-xx,0.5)*sigma[N:2*N,0:N])

sigma[0:N,0:N]=-(np.heaviside(xx-yy,0.5)*sigma[N:2*N,0:N]+np.heaviside(yy-xx,0.5)*sigma[0:N,N:2*N])

ginv_up=g_bare_setup(h1,h2,rho_up)
ginv_down=g_bare_setup(h1,h2,rho_down)


Gnew_up=solve_dyson(ginv_up,sigma,Dt,N)
Gnew_down=solve_dyson(ginv_down,sigma,Dt,N)
nnew_up=np.diagonal(-1j*Gnew_up[0:N,N:2*N])
nnew_down=np.diagonal(-1j*Gnew_down[0:N,N:2*N])
Gold=np.zeros((2*N,2*N))
sigmaHF_up=np.zeros((2*N,2*N))
sigmaHF_down=np.zeros((2*N,2*N))

sigmaHF_up[0:N,0:N]=-U/Dt*nnew_down*np.identity(N)
sigmaHF_up[N:2*N,N:2*N]=U/Dt*nnew_down*np.identity(N)

sigmaHF_down[0:N,0:N]=-U/Dt*nnew_up*np.identity(N)
sigmaHF_down[N:2*N,N:2*N]=U/Dt*nnew_up*np.identity(N)

diff=100.*np.ones(N)
thres=1e-10*np.ones(N)
i=0
start=time.time()

"""Implement a self consistent while loop to evaluate the HF correction"""

while np.greater(diff,thres).all() :
            Gold_up=Gnew_up.copy()
            nold_up=np.diagonal(-1j*Gold_up[0:N,N:2*N])
            
            Gnew_up=solve_dyson(ginv_up,sigmaHF_up+sigma,Dt,N)
            Gnew_down=solve_dyson(ginv_down,sigmaHF_down+sigma,Dt,N)
       
            nnew_down=np.diagonal(-1j*Gnew_down[0:N,N:2*N])
            nnew_up=np.diagonal(-1j*Gnew_up[0:N,N:2*N])
            
            sigmaHF_up[0:N,0:N]=-U/Dt*nnew_down*np.identity(N)
            sigmaHF_up[N:2*N,N:2*N]=U/Dt*nnew_down*np.identity(N)
            sigmaHF_down[0:N,0:N]=-U/Dt*nnew_up*np.identity(N)
            sigmaHF_down[N:2*N,N:2*N]=U/Dt*nnew_up*np.identity(N)
            i=i+1
            diff=abs((nold_up-nnew_up)/nold_up)
            print('Iteration', i, 'complete.', diff)
            
    
#np.save('testup4.npy', nnew_up)
#np.save('testdown4.npy', nnew_down)          
stop=time.time()
tot=stop-start
print('Process completed in', tot, 'seconds.')
print(x1, y1)
x1vec=x1*np.ones(N)
y1vec=y1*np.ones(N)
plt.title(r"$\Gamma=1, U=4, \epsilon_d=\frac{U}{2}$")
ax.plot(x,nnew_down,label=r'$n_{\downarrow},n_{\uparrow}$')
#ax.plot(x,nnew_up,label=r'$n_{\uparrow}$',linestyle='dashdot')
ax.plot(x,x1vec, label= r'analytic, steady state',linestyle='dashdot')

plt.ylabel(r'$n$')
plt.xlabel(r'$t\,(\Gamma^{-1})$')
ax.legend()
ax.grid()
plt.show()
fig.savefig('HF7.PDF')

