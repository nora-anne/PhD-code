import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import optimize
from astropy import units as u
from astropy import constants as c

def Laplace_coeff(alph,s,j):
    """
    Laplace coefficient (Equation 6.67 of Murray & Dermott 1999).
    s is a positive half-integer, j is an integer.
    Unitless.
    """
    x = np.linspace(0,2*np.pi,num=100)
    integrand = np.cos(j*x)/(1-2*alph*np.cos(x)+alph**2)**s
    xspline = UnivariateSpline(x,integrand,s=0)
    return xspline.integral(0,2*np.pi)/np.pi

def g_LL(alph,nb,nc,m_b,m_c,Mstar):
    """
    Gives fundamental secular frequence from linear theory (Omdot or -pomdot).
    In units of nb/nc.
    nb/nc must have same units.
    m_b/m_c/Mstar must have same units.
    """
    b32_1 = Laplace_coeff(alph,3/2,1)
    return -.25*b32_1*alph*(nb*m_c*alph/(Mstar+m_b) + nc*m_b/(Mstar+m_c))

def B(a,ab,ac,Mstar,mb,mc):
    """
    Equation 7.57 from Murray and Dermott.
    a/ab/ac in AU.
    Mstar/mb/mc in MJup.
    Returns in rad per day (with astropy units).
    """
    n = (np.sqrt(c.G*(Mstar*u.Mjup)/(a*u.AU)**3)).to(u.day**-1)
    alph01 = a/ab
    alph02 = a/ac
    alph10 = ab/a
    alph20 = ac/a
    if a<ab:
        return (-u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph01**2*Laplace_coeff(alph01,3/2,1)+
                                    mc*u.Mjup*alph02**2*Laplace_coeff(alph02,3/2,1))).to(u.rad/u.day)
    if a>ab and a<ac:
        return (-u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph10*Laplace_coeff(alph10,3/2,1)+
                                   mc*u.Mjup*alph02**2*Laplace_coeff(alph02,3/2,1))).to(u.rad/u.day)
    if a>ac:
        return (-u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph10*Laplace_coeff(alph10,3/2,1)+
                                   mc*u.Mjup*alph20*Laplace_coeff(alph20,3/2,1))).to(u.rad/u.day)

def B_j(a,aj,Mstar,mj):
    """
    Equation 7.58 from Murray and Dermott.
    a/ab/ac in AU.
    Mstar/mb/mc in MJup.
    Returns in rad per day (with astropy units).
    """
    n = (np.sqrt(c.G*(Mstar*u.Mjup)/(a*u.AU)**3)).to(u.day**-1)
    alph01 = a/aj
    alph10 = aj/a
    if a<aj:
        return u.rad*(n/(4*Mstar*u.Mjup)*(mj*u.Mjup*alph01**2*
                                    Laplace_coeff(alph01,3/2,1))).to(u.day**-1)
    if a>aj:
        return u.rad*(n/(4*Mstar*u.Mjup)*(mj*u.Mjup*alph10*
                                    Laplace_coeff(alph10,3/2,1))).to(u.day**-1)

def B_jj(aj,ak,Mstar,mj,mk):
    """
    Equation 7.11 from Murray and Dermott.
    a/ab/ac in AU.
    Mstar/mb/mc in MJup.
    Returns in rad per day (with astropy units).
    """
    nj = (np.sqrt(c.G*(Mstar*u.Mjup)/(aj*u.AU)**3)).to(u.day**-1)
    if ak<aj:
        alpha = ak/aj
        alphabar = 1
    if ak>aj:
        alpha = aj/ak
        alphabar = aj/ak
    return u.rad*(-nj*1/4*(mk/(Mstar+mj))*alpha*alphabar*Laplace_coeff(alpha,3/2,1)).to(u.day**-1)

def i1i2calc(minc,Mstar,mb,mc,PR,eb,ec):
    def iprime_finder(inc):
        iprime = np.arcsin(np.sin(inc)*mb/mc*PR**(-1/3)
                       *((mb+Mstar)/(mc+Mstar))**(1/6)
                           * np.sqrt((1-eb**2)/(1-ec**2)))
        return iprime - (minc-inc)

    iprime = minc - optimize.fsolve(iprime_finder,.5*minc)[0]

    return minc-iprime,iprime
    
