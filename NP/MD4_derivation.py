import numpy as np
from NP.functions import Laplace_coeff

def Omper_instant(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega period in same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Instantaneous Omega period at a given point in time.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = np.cos(2*omegab)
    cos3 = -np.cos(omegac+omegab)
    cos4 = np.cos(2*omegac)
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    Omdot = .5*(Omdot_c+Omdot_b)
    if np.isnan(Omdot_c):
        return 2*np.pi/abs(Omdot_b)
    if np.isnan(Omdot_b):
        return 2*np.pi/abs(Omdot_c)
    else:
        return 2*np.pi/abs(Omdot)

def Omdot_instant(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Instantaneous Omega dot at a given point in time.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = np.cos(2*omegab)
    cos3 = -np.cos(omegac+omegab)
    cos4 = np.cos(2*omegac)
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    Omdot = .5*(Omdot_b+Omdot_c)
    if np.isnan(Omdot_b) or np.isinf(Omdot_b):
        return Omdot_c
    if np.isnan(Omdot_c) or np.isinf(Omdot_b):
        return Omdot_b
    else:
        return Omdot

def Omdot_b_instant(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot of inner planet in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Instantaneous Omega dot at a given point in time.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = np.cos(2*omegab)
    cos3 = -np.cos(omegac+omegab)
    cos4 = np.cos(2*omegac)
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    return Omdot_b

def Omdot_c_instant(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot of outer planet in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Instantaneous Omega dot at a given point in time.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = np.cos(2*omegab)
    cos3 = -np.cos(omegac+omegab)
    cos4 = np.cos(2*omegac)
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    return Omdot_c

def Omper(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega period in same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Mean Omega period, removed varying omega terms.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = 0
    cos3 = 0
    cos4 = 0
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    Omdot = .5*(Omdot_c+Omdot_b)
    if np.isnan(Omdot_c):
        return 2*np.pi/abs(Omdot_b)
    if np.isnan(Omdot_b):
        return 2*np.pi/abs(Omdot_c)
    else:
        return 2*np.pi/abs(Omdot)

def Omdot(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Mean Omega dot, removed varying omega terms.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = 0
    cos3 = 0
    cos4 = 0
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    Omdot = .5*(Omdot_b+Omdot_c)
    if np.isnan(Omdot_b) or np.isinf(Omdot_b):
        return Omdot_c
    if np.isnan(Omdot_c) or np.isinf(Omdot_b):
        return Omdot_b
    else:
        return Omdot

def Omdot_b(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot of inner planet in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Mean Omega dot, removed varying omega terms.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = 0
    cos3 = 0
    cos4 = 0
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_b = m_c*nb*alph / (4*(m_b+Mstar)*sb*np.sqrt(1-eb**2))  
    B_b = 2*sb*f3+2*sb*(eb**2+ec**2)*f7+4*sb**3*f8+2*sb*sc**2*f9 + \
          (2*sb*f13-sc*f22-sc*f23)*eb*ec*cos1 - \
          sc*f14-(eb**2+ec**2)*sc*f15-(3*sb**2*sc+sc**3)*f16 + \
          (2*sb*f18-sc*f21)*eb**2*cos2 + \
          (2*sb*f19-sc*f24)*eb*ec*cos3 + \
          (2*sb*f20-sc*f25)*ec**2*cos4 + 2*sb*sc**2*f26
    Omdot_b = A_b*B_b
    return Omdot_b

def Omdot_c(alph,b_period,m_b,m_c,Mstar,eb,ec,ib,ic,omegab=np.nan,omegac=np.nan):
    """
    Gives Omega dot of outer planet in rad per same units as b_period.
    m_b/m_c/Mstar must have same units.
    ib/ic/omegab/omegac in rads.
    Mean Omega dot, removed varying omega terms.
    """
    sc = np.sin(ic/2)
    sb = np.sin(ib/2)
    cos1 = -np.cos(omegac-omegab)
    cos2 = 0
    cos3 = 0
    cos4 = 0
    if np.isnan(omegab):
        cos1 = 0
        cos2 = 0
        cos3 = 0
    if np.isnan(omegac):
        cos1 = 0
        cos4 = 0
        cos3 = 0        
    nb = 2*np.pi/b_period
    nc = nb * (alph**3*(m_c+Mstar)/(m_b+Mstar))**.5
    b32_1 = Laplace_coeff(alph,3/2,1)
    b52_0 = Laplace_coeff(alph,5/2,0)
    b52_1 = Laplace_coeff(alph,5/2,1)
    b52_2 = Laplace_coeff(alph,5/2,2)
    b52_3 = Laplace_coeff(alph,5/2,3)
    b72_0 = Laplace_coeff(alph,7/2,0)
    b72_1 = Laplace_coeff(alph,7/2,1)
    b72_2 = Laplace_coeff(alph,7/2,2)
    b72_3 = Laplace_coeff(alph,7/2,3)
    b72_4 = Laplace_coeff(alph,7/2,4)
    f3 = -1/2*alph*b32_1
    f7 = 1/16*(-4*alph*b32_1+30*alph**3*b52_1-12*alph**2*(b52_0+b52_2)
               -15/2*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f8 = 3/4*alph**2*(1/2*b52_2+b52_0)
    f9 = 1/4*alph*(2*b32_1+3*alph*b52_2+15*alph*b52_0)
    f13 = 1/8*(6*alph**2*(b52_3+3*b52_1)-15*alph**3*(b52_2+b52_0)+
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+4)*b72_2
                             -12*alph*b72_1+(4*alph**2+3)*b72_0))
    f22 = 1/4*(-12*alph**2*b52_1+15*alph**3*b52_0-
               15/4*alph**3*(2*b72_2-8*alph*b72_1+(4*alph**2+2)*b72_0))
    f23 = 1/4*(-6*alph**2*(b52_1+b52_3)+15*alph**3*b52_2-
               15/4*alph**3*(b72_4-4*alph*b72_3+(4*alph**2+2)*b72_2
                             -4*alph*b72_1+b72_0))
    f14 = alph*b32_1
    f15 = 1/4*(2*alph*b32_1-15*alph**3*b52_1+6*alph**2*(b52_0+b52_2)+
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f16 = -alph*(1/2*b32_1+3*alph*b52_0+3/2*alph*b52_2)
    f18 = 1/16*(12*alph*b32_1-27*alph**3*b52_1+12*alph**2*(b52_2+b52_0)+
                15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f21 = 1/8*(-12*alph*b32_1+27*alph**3*b52_1-12*alph**2*(b52_2+b52_0)-
               15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-4*alph*b72_0))
    f19 = 1/2*f22
    f24 = -f22
    f20 = 1/16*(15/4*alph**3*(b72_3-4*alph*b72_2+(4*alph**2+3)*b72_1-
                              4*alph*b72_0-4/5*b52_1))
    f25 = -2*f20
    f26 = 1/2*alph*(b32_1+3/2*alph*b52_0+3*alph*b52_2)
    A_c = nc*m_b/(4*(m_c+Mstar)*np.sqrt(1-ec**2)*sc)
    B_c = 2*sc*f3+(2*eb**2*sc+2*ec**2*sc)*f7+4*sc**3*f8+2*sb**2*sc*f9 + \
          (2*sc*f13-sb*f22-sb*f23)*eb*ec*cos1 - \
          sb*f14-(eb**2+ec**2)*sb*f15-(sb**3+3*sb*sc**2)*f16 + \
          (2*sc*f18-sb*f21)*eb**2*cos2 + \
          (2*sc*f19-sb*f24)*eb*ec*cos3 + \
          (2*sc*f20-sb*f25)*ec**2*cos4+2*sb**2*sc*f26
    Omdot_c = A_c*B_c
    return Omdot_c
