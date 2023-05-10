import numpy as np
from scipy.interpolate import UnivariateSpline
from NP.functions import Laplace_coeff
from scipy.optimize import fsolve
from astropy import units as u
from astropy import constants as c
from Functions.orbits import PtoA,AtoP

def A_jj(j,M,P):
    """
    Equation 7.132 of Murray & Dermott, neglecting non-point source effects.
    j is desired index.
    P is array of periods (with astropy units).
    M is array of masses (with astropy units).
    """
    SMA = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    c1 = nj/4
    c2 = 0
    for k in range(1,1+len(P)):
        if k!=j:
            alpha_jk = min(SMA[j-1]/SMA[k-1],SMA[k-1]/SMA[j-1])
            alphabar_jk = min(SMA[j-1]/SMA[k-1],1)
            dum = M[k]/(M[0]+M[j])*alpha_jk*alphabar_jk*Laplace_coeff(
                alpha_jk,3/2,1)
            c2 += dum
    return c1*c2

def A_jk(j,k,M,P):
    """
    Equation 7.133 of Murray & Dermott, neglecting non-point source effects.
    j,k are desired indices (j!=k).
    P is array of periods (with astropy units).
    M is array of masses (with astropy units).
    """
    SMA = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    alpha_jk = min(SMA[j-1]/SMA[k-1],SMA[k-1]/SMA[j-1])
    alphabar_jk = min(SMA[j-1]/SMA[k-1],1)
    return -M[k]/(M[0]+M[j])*nj/4*alpha_jk*alphabar_jk*Laplace_coeff(alpha_jk,3/2,2)

def B_jj(j,M,P):
    """
    Equation 7.134 of Murray & Dermott, neglecting non-point source effects.
    j is desired index.
    P is array of periods (with astropy units).
    M is array of masses (with astropy units).
    """
    SMA = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    c1 = -nj/4
    c2 = 0
    for k in range(1,1+len(P)):
        if k!=j:
            alpha_jk = min(SMA[j-1]/SMA[k-1],SMA[k-1]/SMA[j-1])
            alphabar_jk = min(SMA[j-1]/SMA[k-1],1)
            dum = M[k]/(M[0]+M[j])*alpha_jk*alphabar_jk*Laplace_coeff(
                alpha_jk,3/2,1)
            c2 += dum
    return c1*c2

def B_jk(j,k,M,P):
    """
    Equation 7.135 of Murray & Dermott, neglecting non-point source effects.
    j,k are desired indices (j!=k).
    P is array of periods (with astropy units).
    M is array of masses (with astropy units).
    """
    SMA = ((c.G*(M[0]+M[1:])/(4*np.pi**2)) * P**2)**(1/3)
    nj = 2*np.pi*u.rad/P[j-1]
    alpha_jk = min(SMA[j-1]/SMA[k-1],SMA[k-1]/SMA[j-1])
    alphabar_jk = min(SMA[j-1]/SMA[k-1],1)
    return M[k]/(M[0]+M[j])*nj/4*alpha_jk*alphabar_jk*Laplace_coeff(alpha_jk,3/2,1)

def A_matrix(M,P):
    """
    M is array of masses (with astropy units).
    P is array of periods (with astropy units).
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
    M is array of masses (with astropy units).
    P is array of periods (with astropy units).
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

def A(SMA,M):
    """
    Equation 7.55 from Murray and Dermott.
    SMA in AU.
    M in MJup (do not include 0 for test particle).
    Returns in rad per day (with astropy units).
    """
    a,ab,ac = SMA
    Mstar,mb,mc = M
    n = (np.sqrt(c.G*(Mstar*u.Mjup)/(a*u.AU)**3)).to(u.day**-1)
    alph01 = a/ab
    alph02 = a/ac
    alph10 = ab/a
    alph20 = ac/a
    if a<ab:
        return (u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph01**2*Laplace_coeff(alph01,3/2,1)+
                                    mc*u.Mjup*alph02**2*Laplace_coeff(alph02,3/2,1))).to(u.rad/u.day)
    if a>ab and a<ac:
        return (u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph10*Laplace_coeff(alph10,3/2,1)+
                                   mc*u.Mjup*alph02**2*Laplace_coeff(alph02,3/2,1))).to(u.rad/u.day)
    if a>ac:
        return (u.rad*n/(4*Mstar*u.Mjup)*(mb*u.Mjup*alph10*Laplace_coeff(alph10,3/2,1)+
                                   mc*u.Mjup*alph20*Laplace_coeff(alph20,3/2,1))).to(u.rad/u.day)


def A_j(a,aj,Mstar,mj):
    """
    Equation 7.56 from Murray and Dermott.
    a/aj in AU.
    Mstar/mj in MJup.
    Returns in rad per day (with astropy units).
    """
    n = (np.sqrt(c.G*(Mstar*u.Mjup)/(a*u.AU)**3)).to(u.day**-1)
    alph01 = a/aj
    alph10 = aj/a
    if a<aj:
        return u.rad*(-n/(4*Mstar*u.Mjup)*(mj*u.Mjup*alph01**2*
                                    Laplace_coeff(alph01,3/2,2))).to(u.day**-1)
    if a>aj:
        return u.rad*(-n/(4*Mstar*u.Mjup)*(mj*u.Mjup*alph10*
                                    Laplace_coeff(alph10,3/2,2))).to(u.day**-1)


def B(SMA,M):
    """
    Equation 7.57 from Murray and Dermott.
    SMA in AU.
    M in MJup (do not include 0 for test particle).
    Returns in rad per day (with astropy units).
    """
    a,ab,ac = SMA
    Mstar,mb,mc = M
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
    a/aj in AU.
    Mstar/mj in MJup.
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

def max_forced_e(M,SMA,E,Pom):
    """
    Returns maximum forced eccentricity over 1e9 orbits for a test particle.
    M/SMA is array of masses/semi-major axes (including test particle)
    E/Pom is array of eccentricites/pomegas (not including test particle) at t=0.
    Input arrays have astropy units.
    """

    #get arrays without test particle included
    M_withmass = M[M!=0]
    SMA_withmass = SMA[M[1:]!=0]
    P = AtoP(M,SMA)
    P_withmass = AtoP(M_withmass,SMA_withmass)
    Ptp = P[M[1:]==0]
    SMAtp = SMA[M[1:]==0]

    Amat = A_matrix(M_withmass,P_withmass)

    w,v = np.linalg.eig(Amat)
    g1,g2 = w

    #unscaled eigenvector components
    e11u = v[0,0]
    e21u = v[1,0]
    e12u = v[0,1]
    e22u = v[1,1]

    #solve for boundary conditions at t=0
    h1,h2 = E*np.sin(Pom)
    k1,k2 = E*np.cos(Pom)

    def equations(p):
        S1,S2,b1,b2 = p
        fun1 = S1*e11u*np.sin(b1) + S2*e12u*np.sin(b2) - h1
        fun2 = S1*e11u*np.cos(b1) + S2*e12u*np.cos(b2) - k1
        fun3 = S1*e21u*np.sin(b1) + S2*e22u*np.sin(b2) - h2
        fun4 = S1*e21u*np.cos(b1) + S2*e22u*np.cos(b2) - k2
        return (fun1,fun2,fun3,fun4)


    S1,S2,b1,b2 = fsolve(equations,[0,0,0,0])
    b1 *= u.rad
    b2 *= u.rad

    #scaled eigenvector components
    e11 = S1*e11u
    e12 = S2*e12u
    e21 = S1*e21u
    e22 = S2*e22u

    def e_forced(t,SMA,M,g1,g2):
        SMA_AU = SMA.to_value(u.AU)
        M_Mjup = M.to_value(u.Mjup)
        Ak = A(SMA_AU,M_Mjup)
        nu1 = A_j(SMA_AU[0],SMA_AU[1],M_Mjup[0],M_Mjup[1])*e11 + \
              A_j(SMA_AU[0],SMA_AU[2],M_Mjup[0],M_Mjup[2])*e21
        nu2 = A_j(SMA_AU[0],SMA_AU[1],M_Mjup[0],M_Mjup[1])*e12 + \
              A_j(SMA_AU[0],SMA_AU[2],M_Mjup[0],M_Mjup[2])*e22

        h0 = -nu1/(Ak-g1)*np.sin(g1*t+b1) - nu2/(Ak-g2)*np.sin(g2*t+b2)
        k0 = -nu1/(Ak-g1)*np.cos(g1*t+b1) - nu2/(Ak-g2)*np.cos(g2*t+b2)

        return np.sqrt(h0**2+k0**2)

    t1 = ((b1-b2)/(g2-g1)).to(Ptp.unit)
    t2 = ((b1-b2+np.pi*u.rad)/(g2-g1)).to(Ptp.unit)
    e1 = e_forced(t1,SMA,M_withmass,g1,g2)
    e2 = e_forced(t2,SMA,M_withmass,g1,g2)

    e_max = max(e1,e2)

    return e_max

def max_inner_e(M,SMA,E,Pom):
    """
    Returns maximum eccentricity over 1e9 orbits for the inner planet.
    M/SMA is array of masses/semi-major axes 
    E/Pom is array of eccentricites/pomegas at t=0.
    Input arrays have astropy units.
    """

    P = AtoP(M,SMA)
    
    Amat = A_matrix(M,P).to(u.deg/u.yr)

    w,v = np.linalg.eig(Amat)
    g1,g2,g3 = w

    #unscaled eigenvector components
    e11u = v[0,0]
    e21u = v[1,0]
    e31u = v[2,0]
    e12u = v[0,1]
    e22u = v[1,1]
    e32u = v[2,1]
    e13u = v[0,2]
    e23u = v[1,2]
    e33u = v[2,2]

    #solve for boundary conditions at t=0
    h1,h2,h3 = E *np.sin(Pom)
    k1,k2,k3 = E *np.cos(Pom)

    def equations(p):
        S1,S2,S3,b1,b2,b3 = p
        fun1 = S1*e11u*np.sin(b1) + S2*e12u*np.sin(b2) + S3*e13u*np.sin(b3) - h1
        fun2 = S1*e11u*np.cos(b1) + S2*e12u*np.cos(b2) + S3*e13u*np.cos(b3) - k1
        fun3 = S1*e21u*np.sin(b1) + S2*e22u*np.sin(b2) + S3*e23u*np.sin(b3) - h2
        fun4 = S1*e21u*np.cos(b1) + S2*e22u*np.cos(b2) + S3*e23u*np.cos(b3) - k2
        fun5 = S1*e31u*np.sin(b1) + S2*e32u*np.sin(b2) + S3*e33u*np.sin(b3) - h3
        fun6 = S1*e31u*np.cos(b1) + S2*e32u*np.cos(b2) + S3*e33u*np.cos(b3) - k3
        return (fun1,fun2,fun3,fun4,fun5,fun6)

    S1,S2,S3,b1,b2,b3 = fsolve(equations,[0,0,0,0,0,0])
    coeffsum = np.sum([S1,S2,S3,b1,b2,b3])
    if coeffsum<=0:
        S1,S2,S3,b1,b2,b3 = fsolve(equations,[np.pi,np.pi,np.pi,np.pi,np.pi,np.pi])
    b1 *= u.rad
    b2 *= u.rad
    b3 *= u.rad

    #scaled eigenvector components
    e11 = S1*e11u
    e12 = S2*e12u
    e13 = S3*e13u
    e21 = S1*e21u
    e22 = S2*e22u
    e23 = S3*e23u
    e31 = S1*e31u
    e32 = S2*e32u
    e33 = S3*e33u

    def e_earth_overtime(t,g1,g2,g3):
        hE = e11*np.sin(g1*t+b1)+e12*np.sin(g2*t+b2)+e13*np.sin(g3*t+b3)
        kE = e11*np.cos(g1*t+b1)+e12*np.cos(g2*t+b2)+e13*np.cos(g3*t+b3)

        return np.sqrt(hE**2+kE**2)

    ts1 = np.linspace(0,1e9,num=1000)
    es1 = e_earth_overtime(ts1*u.yr,g1,g2,g3)
    ts2 = np.linspace(ts1[np.argmax(es1)]-5e6,ts1[np.argmax(es1)]+5e6,num=10000)
    es2 = e_earth_overtime(ts2*u.yr,g1,g2,g3)
    e_max = max(max(es2),max(es1))

    return e_max

def max_e_3planets(M,SMA,E,Pom):
    """
    Returns maximum eccentricity over 1e9 orbits for all 3 planets.
    Also returns eccentricity of other 2 planets at the point in time one is at max.
    Output array is 3x3, where each row is a time with the max e first.
    M/SMA is array of masses/semi-major axes 
    E/Pom is array of eccentricites/pomegas at t=0.
    Input arrays have astropy units.
    """

    P = AtoP(M,SMA)
    
    Amat = A_matrix(M,P).to(u.deg/u.yr)

    w,v = np.linalg.eig(Amat)
    g1,g2,g3 = w

    #unscaled eigenvector components
    e11u = v[0,0]
    e21u = v[1,0]
    e31u = v[2,0]
    e12u = v[0,1]
    e22u = v[1,1]
    e32u = v[2,1]
    e13u = v[0,2]
    e23u = v[1,2]
    e33u = v[2,2]

    #solve for boundary conditions at t=0
    h1,h2,h3 = E *np.sin(Pom)
    k1,k2,k3 = E *np.cos(Pom)

    def equations(p):
        S1,S2,S3,b1,b2,b3 = p
        fun1 = S1*e11u*np.sin(b1) + S2*e12u*np.sin(b2) + S3*e13u*np.sin(b3) - h1
        fun2 = S1*e11u*np.cos(b1) + S2*e12u*np.cos(b2) + S3*e13u*np.cos(b3) - k1
        fun3 = S1*e21u*np.sin(b1) + S2*e22u*np.sin(b2) + S3*e23u*np.sin(b3) - h2
        fun4 = S1*e21u*np.cos(b1) + S2*e22u*np.cos(b2) + S3*e23u*np.cos(b3) - k2
        fun5 = S1*e31u*np.sin(b1) + S2*e32u*np.sin(b2) + S3*e33u*np.sin(b3) - h3
        fun6 = S1*e31u*np.cos(b1) + S2*e32u*np.cos(b2) + S3*e33u*np.cos(b3) - k3
        return (fun1,fun2,fun3,fun4,fun5,fun6)

    S1,S2,S3,b1,b2,b3 = fsolve(equations,[0,0,0,0,0,0])
    coeffsum = np.sum([S1,S2,S3,b1,b2,b3])
    if coeffsum<=0:
        S1,S2,S3,b1,b2,b3 = fsolve(equations,[np.pi,np.pi,np.pi,np.pi,np.pi,np.pi])
    b1 *= u.rad
    b2 *= u.rad
    b3 *= u.rad

    #scaled eigenvector components
    e11 = S1*e11u
    e12 = S2*e12u
    e13 = S3*e13u
    e21 = S1*e21u
    e22 = S2*e22u
    e23 = S3*e23u
    e31 = S1*e31u
    e32 = S2*e32u
    e33 = S3*e33u

    def e_overtime(t,ec1,ec2,ec3,b1,b2,b3,g1,g2,g3):
        h_t = ec1*np.sin(g1*t+b1)+ec2*np.sin(g2*t+b2)+ec3*np.sin(g3*t+b3)
        k_t = ec1*np.cos(g1*t+b1)+ec2*np.cos(g2*t+b2)+ec3*np.cos(g3*t+b3)

        return np.sqrt(h_t**2+k_t**2)

    ts0 = np.linspace(0,1e9,num=1000)

    #planet 1
    es1_0 = e_overtime(ts0*u.yr,e11,e12,e13,b1,b2,b3,g1,g2,g3)
    ts1 = np.linspace(ts0[np.argmax(es1_0)]-5e6,
                      ts0[np.argmax(es1_0)]+5e6,num=10000)
    es1 = e_overtime(ts1*u.yr,e11,e12,e13,b1,b2,b3,g1,g2,g3)
    e1_max = max(es1)
    e2_te1max = e_overtime(ts1[np.argmax(es1)]*u.yr,e21,e22,e23,b1,b2,b3,g1,g2,g3)
    e3_te1max = e_overtime(ts1[np.argmax(es1)]*u.yr,e31,e32,e33,b1,b2,b3,g1,g2,g3)

    #planet 2
    es2_0 = e_overtime(ts0*u.yr,e21,e22,e23,b1,b2,b3,g1,g2,g3)
    ts2 = np.linspace(ts0[np.argmax(es2_0)]-5e6,
                      ts0[np.argmax(es2_0)]+5e6,num=10000)
    es2 = e_overtime(ts2*u.yr,e21,e22,e23,b1,b2,b3,g1,g2,g3)
    e2_max = max(es2)
    e1_te2max = e_overtime(ts2[np.argmax(es2)]*u.yr,e11,e12,e13,b1,b2,b3,g1,g2,g3)
    e3_te2max = e_overtime(ts2[np.argmax(es2)]*u.yr,e31,e32,e33,b1,b2,b3,g1,g2,g3)

    #planet 3
    es3_0 = e_overtime(ts0*u.yr,e31,e32,e33,b1,b2,b3,g1,g2,g3)
    ts3 = np.linspace(ts0[np.argmax(es3_0)]-5e6,
                      ts0[np.argmax(es3_0)]+5e6,num=10000)
    es3 = e_overtime(ts3*u.yr,e31,e32,e33,b1,b2,b3,g1,g2,g3)
    e3_max = max(es3)
    e1_te3max = e_overtime(ts3[np.argmax(es3)]*u.yr,e11,e12,e13,b1,b2,b3,g1,g2,g3)
    e2_te3max = e_overtime(ts3[np.argmax(es3)]*u.yr,e21,e22,e23,b1,b2,b3,g1,g2,g3)

    return np.array([[e1_max,e2_te1max,e3_te1max],
                    [e2_max,e1_te2max,e3_te2max],
                    [e3_max,e1_te3max,e2_te3max]])

def e_tmax_3planets(M,SMA,E,Pom,tmax):
    """
    Returns eccentricity over tmax years for all 3 planets.
    M/SMA is array of masses/semi-major axes 
    E/Pom is array of eccentricites/pomegas at t=0.
    Input arrays have astropy units.
    """

    P = AtoP(M,SMA)
    
    Amat = A_matrix(M,P).to(u.deg/u.yr)

    w,v = np.linalg.eig(Amat)
    g1,g2,g3 = w

    #unscaled eigenvector components
    e11u = v[0,0]
    e21u = v[1,0]
    e31u = v[2,0]
    e12u = v[0,1]
    e22u = v[1,1]
    e32u = v[2,1]
    e13u = v[0,2]
    e23u = v[1,2]
    e33u = v[2,2]

    #solve for boundary conditions at t=0
    h1,h2,h3 = E *np.sin(Pom)
    k1,k2,k3 = E *np.cos(Pom)

    def equations(p):
        S1,S2,S3,b1,b2,b3 = p
        fun1 = S1*e11u*np.sin(b1) + S2*e12u*np.sin(b2) + S3*e13u*np.sin(b3) - h1
        fun2 = S1*e11u*np.cos(b1) + S2*e12u*np.cos(b2) + S3*e13u*np.cos(b3) - k1
        fun3 = S1*e21u*np.sin(b1) + S2*e22u*np.sin(b2) + S3*e23u*np.sin(b3) - h2
        fun4 = S1*e21u*np.cos(b1) + S2*e22u*np.cos(b2) + S3*e23u*np.cos(b3) - k2
        fun5 = S1*e31u*np.sin(b1) + S2*e32u*np.sin(b2) + S3*e33u*np.sin(b3) - h3
        fun6 = S1*e31u*np.cos(b1) + S2*e32u*np.cos(b2) + S3*e33u*np.cos(b3) - k3
        return (fun1,fun2,fun3,fun4,fun5,fun6)

    S1,S2,S3,b1,b2,b3 = fsolve(equations,[0,0,0,0,0,0])
    coeffsum = np.sum([S1,S2,S3,b1,b2,b3])
    if coeffsum<=0:
        S1,S2,S3,b1,b2,b3 = fsolve(equations,[np.pi,np.pi,np.pi,np.pi,np.pi,np.pi])
    b1 *= u.rad
    b2 *= u.rad
    b3 *= u.rad

    #scaled eigenvector components
    e11 = S1*e11u
    e12 = S2*e12u
    e13 = S3*e13u
    e21 = S1*e21u
    e22 = S2*e22u
    e23 = S3*e23u
    e31 = S1*e31u
    e32 = S2*e32u
    e33 = S3*e33u

    def e_overtime(t,ec1,ec2,ec3,b1,b2,b3,g1,g2,g3):
        h_t = ec1*np.sin(g1*t+b1)+ec2*np.sin(g2*t+b2)+ec3*np.sin(g3*t+b3)
        k_t = ec1*np.cos(g1*t+b1)+ec2*np.cos(g2*t+b2)+ec3*np.cos(g3*t+b3)

        return np.sqrt(h_t**2+k_t**2).value

    ts0 = np.linspace(0,tmax,num=10000)

    #planet 1
    es1 = e_overtime(ts0*u.yr,e11,e12,e13,b1,b2,b3,g1,g2,g3)
    
    #planet 2
    es2 = e_overtime(ts0*u.yr,e21,e22,e23,b1,b2,b3,g1,g2,g3)
    
    #planet 3
    es3 = e_overtime(ts0*u.yr,e31,e32,e33,b1,b2,b3,g1,g2,g3)
    
    return es1,es2,es3

def max_forced_i(M,SMA,I,Om):
    """
    Returns maximum forced inclination over 1e9 orbits for a test particle.
    M/SMA is array of masses/semi-major axes (including test particle)
    I/Om is array of inclinations/Omegas (not including test particle) at t=0.
    Input arrays have astropy units.
    """

    #get arrays without test particle included
    M_withmass = M[M!=0]
    SMA_withmass = SMA[M[1:]!=0]
    P = AtoP(M,SMA)
    P_withmass = AtoP(M_withmass,SMA_withmass)
    Ptp = P[M[1:]==0]
    SMAtp = SMA[M[1:]==0]

    Bmat = B_matrix(M_withmass,P_withmass)

    w,v = np.linalg.eig(Bmat)
    f1,f2 = w

    #unscaled eigenvector components
    i11u = v[0,0]
    i21u = v[1,0]
    i12u = v[0,1]
    i22u = v[1,1]

    #solve for boundary conditions at t=0
    p1,p2 = I*np.sin(Om)
    q1,q2 = I*np.cos(Om)

    def equations(p):
        T1,T2,g1,g2 = p
        fun1 = T1*i11u*np.sin(g1) + T2*i12u*np.sin(g2) - p1
        fun2 = T1*i11u*np.cos(g1) + T2*i12u*np.cos(g2) - q1
        fun3 = T1*i21u*np.sin(g1) + T2*i22u*np.sin(g2) - p2
        fun4 = T1*i21u*np.cos(g1) + T2*i22u*np.cos(g2) - q2
        return (fun1,fun2,fun3,fun4)

    T1,T2,gam1,gam2 = fsolve(equations,[1,1,3,3])
    gam1 *= u.rad
    gam2 *= u.rad

    #scaled eigenvector components
    i11 = T1*i11u
    i12 = T2*i12u
    i21 = T1*i21u
    i22 = T2*i22u

    def i_forced(t,SMA,M,f1,f2):
        SMA_AU = SMA.to_value(u.AU)
        M_Mjup = M.to_value(u.Mjup)
        Bk = B(SMA_AU,M_Mjup)
        mu1 = B_j(SMA_AU[0],SMA_AU[1],M_Mjup[0],M_Mjup[1])*i11 + \
              B_j(SMA_AU[0],SMA_AU[2],M_Mjup[0],M_Mjup[2])*i21
        mu2 = B_j(SMA_AU[0],SMA_AU[1],M_Mjup[0],M_Mjup[1])*i12 + \
              B_j(SMA_AU[0],SMA_AU[2],M_Mjup[0],M_Mjup[2])*i22

        p0 = -mu1/(Bk-f1)*np.sin(f1*t+gam1) - mu2/(Bk-f2)*np.sin(f2*t+gam2)
        q0 = -mu1/(Bk-f1)*np.cos(f1*t+gam1) - mu2/(Bk-f2)*np.cos(f2*t+gam2)

        return np.sqrt(p0**2+q0**2)

    t1 = ((gam1-gam2)/(f2-f1)).to(Ptp.unit)
    t2 = ((gam1-gam2+np.pi*u.rad)/(f2-f1)).to(Ptp.unit)
    i1 = i_forced(t1,SMA,M_withmass,f1,f2)
    i2 = i_forced(t2,SMA,M_withmass,f1,f2)

    i_max = max(i1,i2)

    return i_max
