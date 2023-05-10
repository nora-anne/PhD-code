import numpy as np
import astropy.units as u
import astropy.constants as c

def PtoA(M,P):
    """
    For a given set of periods and masses, calculates semi-major axes.
    Using Kepler's Third Law.
    Input with any astropy units, output with AU astropy units.
    """
    Mstar = M[0]
    A = np.empty(len(P))
    for i,Mp in enumerate(M[1:]):
        A[i] = ((P[i]**2*c.G*(Mstar+Mp)/(4*np.pi**2))**(1/3)).to_value(u.AU)

    return A*u.AU

def AtoP(M,A):
    """
    For a given set of semi-major axes and masses, calculates periods.
    Using Kepler's Third Law.
    Input with any astropy units, output with day astropy units.
    """
    Mstar = M[0]
    P = np.empty(len(A))
    for i,Mp in enumerate(M[1:]):
        P[i] = ((A[i]**3/(c.G*(Mstar+Mp)/(4*np.pi**2)))**(1/2)).to_value(u.day)

    return P*u.day
