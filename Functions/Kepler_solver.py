import numpy as np

def get_E(e,M):
    """
    For a given eccentricity and mean anomaly, calculates eccentric anomaly.
    """
    Eguess = M
    converge = False
    while converge==False:
        g = Eguess - e*np.sin(Eguess) - M
        g_p = 1 - e*np.cos(Eguess)
        E_iter = Eguess - g/g_p
        if abs(E_iter-Eguess) < 1e-12:
            converge = True
        Eguess = E_iter

    return E_iter

def get_E_niter(e,M,n):
    """
    For a given eccentricity and mean anomaly, calculates eccentric anomaly.
    Iterates for n steps.
    """
    Eguess = M
    niter = 0
    while niter<n:
        g = Eguess - e*np.sin(Eguess) - M
        g_p = 1 - e*np.cos(Eguess)
        E_iter = Eguess - g/g_p
        tol = abs(E_iter-Eguess)
        Eguess = E_iter
        niter += 1

    return E_iter,tol

def get_f(e,M):
    """
    For a given eccentricity and mean anomaly, calculates true anomaly.
    """
    E = get_E(e,M)
    f = 2 * np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    return f

def get_f_niter(e,M,n):
    """
    For a given eccentricity and mean anomaly, calculates true anomaly.
    Uses n steps to iterate for eccentric anomaly.
    """
    E,tol = get_E_niter(e,M,n)
    f = 2 * np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))

    return f,tol
