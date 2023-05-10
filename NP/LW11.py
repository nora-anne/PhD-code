import numpy as np

def gamma(a_b,a_c,m_c,Mstar):
    """
    Equation 15 of Lithwick & Wu 2011.
    Applies only for inner test particle, low alpha.
    a_b/a_c in AU, m_c/Mstar in Mgeo, gives rad/yr.
    """
    G = 0.00011856852799449061 #grav constant in units of AU^3/(Mearth yr^2)
    dummy = G*Mstar/a_b**3
    return 3/4 * m_c/Mstar * (a_b/a_c)**3 * dummy**.5

def pomegadot(a_b,a_c,m_c,Mstar,e_b,i_b):
    """
    Equation 31 of Lithwick & Wu 2011.
    Nonlinear apsidal frequency for inner test particle (for e=i=0 of outer planet).
    a_b/a_c in AU, m_c/Mstar in Mgeo, i_b in rad.
    Gives rad/yr.
    """
    pe = 2*(1-np.sqrt(1-e_b**2)) #eqn 1 LW11
    pi = 4*np.sqrt(1-e_b**2)*np.sin(i_b/2)**2 #eqn 6 LW11
    gam = gamma(a_b,a_c,m_c,Mstar)
    return gam*(1-.5*pe-2*pi)

def Omegadot(a_b,a_c,m_c,Mstar,e_b,i_b):
    """
    Equation 33 of Lithwick & Wu 2011.
    Nonlinear nodal frequency for inner test particle (for e=i=0 of outer planet).
    a_b/a_c in AU, m_c/Mstar in Mgeo, i_b in rad.
    Gives rad/yr.
    """
    pe = 2*(1-np.sqrt(1-e_b**2)) #eqn 1 LW11
    pi = 4*np.sqrt(1-e_b**2)*np.sin(i_b/2)**2 #eqn 6 LW11
    gam = gamma(a_b,a_c,m_c,Mstar)
    return gam*(-1+.5*pi-2*pe)
