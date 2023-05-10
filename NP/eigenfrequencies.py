import numpy as np
from scipy.interpolate import UnivariateSpline
from NP.functions import Laplace_coeff
from scipy.optimize import fsolve
from astropy import units as u
from astropy import constants as c

def A_jj(j,M,P):
    """
    j is desired index.
    P is array of periods.
    M is array of masses.
    """
    A = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    c1 = nj/4
    c2 = 0
    for k in range(1,1+len(P)):
        if k!=j:
            alpha_jk = min(A[j-1]/A[k-1],A[k-1]/A[j-1])
            alphabar_jk = min(A[j-1]/A[k-1],1)
            dum = M[k]/(M[0]+M[j])*alpha_jk*alphabar_jk*Laplace_coeff(
                alpha_jk,3/2,1)
            c2 += dum
    return c1*c2

def A_jk(j,k,M,P):
    """
    j,k are desired indices (j!=k).
    P is array of periods.
    M is array of masses.
    """
    A = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    alpha_jk = min(A[j-1]/A[k-1],A[k-1]/A[j-1])
    alphabar_jk = min(A[j-1]/A[k-1],1)
    return -M[k]/(M[0]+M[j])*nj/4*alpha_jk*alphabar_jk*Laplace_coeff(alpha_jk,3/2,2)

def B_jj(j,M,P):
    """
    j is desired index.
    P is array of periods.
    M is array of masses.
    """
    A = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    c1 = -nj/4
    c2 = 0
    for k in range(1,1+len(P)):
        if k!=j:
            alpha_jk = min(A[j-1]/A[k-1],A[k-1]/A[j-1])
            alphabar_jk = min(A[j-1]/A[k-1],1)
            dum = M[k]/(M[0]+M[j])*alpha_jk*alphabar_jk*Laplace_coeff(
                alpha_jk,3/2,1)
            c2 += dum
    return c1*c2

def B_jk(j,k,M,P):
    """
    j,k are desired indices (j!=k).
    P is array of periods.
    M is array of masses.
    """
    A = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    alpha_jk = min(A[j-1]/A[k-1],A[k-1]/A[j-1])
    alphabar_jk = min(A[j-1]/A[k-1],1)
    return M[k]/(M[0]+M[j])*nj/4*alpha_jk*alphabar_jk*Laplace_coeff(alpha_jk,3/2,1)

def A_matrix(M,P):
    """
    M is array of masses.
    P is array of periods.
    """
    N = len(P)
    Amat = np.empty((N,N))
    for j in range(N):
        for k in range(N):
            if j==k:
                Amat[j,k] = A_jj(j+1,M,P).value
            else:
                Amat[j,k] = A_jk(j+1,k+1,M,P).value

    units = A_jj(1,M,P).unit
    return Amat*units
            
def B_matrix(M,P):
    """
    M is array of masses.
    P is array of periods.
    """
    N = len(P)
    Bmat = np.empty((N,N))
    for j in range(N):
        for k in range(N):
            if j==k:
                Bmat[j,k] = B_jj(j+1,M,P).value
            else:
                Bmat[j,k] = B_jk(j+1,k+1,M,P).value

    units = B_jj(1,M,P).unit
    return Bmat*units

def g_eigenfreqs(M,P):
    """
    Solves for g (eccentricity) frequencies that are eigenvalues of A matrix.
    M is array of the masses (with units), where M[0] = star
    P is array of periods (with units), where P[0] = P1
    Returns g eigenfrequencies with units.
    """
    v = np.linalg.eigvals(A_matrix(M,P))
    return v

def f_eigenfreqs(M,P):
    """
    Solves for f (inclination) frequencies that are eigenvalues of B matrix.
    M is array of the masses (with units), where M[0] = star
    P is array of periods (with units), where P[0] = P1
    Returns f eigenfrequencies with units.
    """
    v = np.linalg.eigvals(B_matrix(M,P)) 
    return v[abs(v)!=np.min(abs(v))]
