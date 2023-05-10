import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from celmech.disturbing_function import get_fg_coeffs
import cmath
import time

def mu(m,Mstar):
    return m*Mstar/(m+Mstar)

def alpha0(j,k,Mstar,m1,m2):
    return ((j-k)/j)**(2/3)*((Mstar+m1)/(Mstar+m2))**(1/3)

def A(j,k,Mstar,m1,m2):
    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)
    alpha00 = alpha0(j,k,Mstar,m1,m2)

    return 3*j*(mu1+mu2)/2*(j/mu2 + (j-k)/(mu1*np.sqrt(alpha00)))

def ftilde(j,k,Mstar,m1,m2):
    f = get_fg_coeffs(j,k)[0]
    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)
    alpha00 = alpha0(j,k,Mstar,m1,m2)

    return np.sqrt((mu1+mu2)/(mu1*np.sqrt(alpha00)))*f

def gtilde(j,k,Mstar,m1,m2):
    g = get_fg_coeffs(j,k)[1]
    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)

    return np.sqrt((mu1+mu2)/mu2)*g

def Ham_eqn19(J,theta,Jstar,sysprops):
    j,k,Mstar,m1,m2 = sysprops

    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)
    alpha00 = alpha0(j,k,Mstar,m1,m2)
    A0 = A(j,k,Mstar,m1,m2)
    f,g = get_fg_coeffs(j,k)
    ftilde0 = ftilde(j,k,Mstar,m1,m2)
    gtilde0 = gtilde(j,k,Mstar,m1,m2)
    epsilon = m1*mu2/(Mstar*(mu1+mu2))
    epstilde = 2*(ftilde0**2+gtilde0**2)**(k/2)*epsilon

    return -1/(2*k**2) *A0 *(J-Jstar)**2 - epstilde*J**(k/2)*np.cos(k*theta)

def J(sysprops,e1,e2,pomega1,pomega2):
    j,k,Mstar,m1,m2 = sysprops

    f,g = get_fg_coeffs(j,k)
    ftilde0 = ftilde(j,k,Mstar,m1,m2)
    gtilde0 = gtilde(j,k,Mstar,m1,m2)

    return (f**2*e1**2+g**2*e2**2+2*f*g*e1*e2*np.cos(pomega2-pomega1))/(
        ftilde0**2+gtilde0**2)

def P(sysprops,P1,P2):
    j,k,Mstar,m1,m2 = sysprops

    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)
    alpha00 = alpha0(j,k,Mstar,m1,m2)

    return ((j-k)/j*(P2/P1) - 1 )/(
        3*(mu1+mu2)/2*(j/mu2+(j-k)/(mu1*np.sqrt(alpha00))))

def Jstar(sysprops,e1,e2,pomega1,pomega2,P1,P2):
    k = sysprops[1]
    Jdum = J(sysprops,e1,e2,pomega1,pomega2)
    Pdum = P(sysprops,P1,P2)

    return Jdum - k*Pdum

def max_Ham_eqn19(J0,Jstar0,sysprops):
    """
    Maximizes equation 19 for J, given Jstar and set of j,k,Mstar,m1,m2.
    J0 is initial guess for maximized J.
    """
    def oppHam(J,theta,Jstar,sysprops):
        return -Ham_eqn19(J,theta,Jstar,sysprops)

    return optimize.minimize(oppHam,J0,args=(0,Jstar0,sysprops))['x'][0]

def Ham_eqn19_Jsolver(J00,J01,Jstar0,sysprops,E):
    """
    Solves equation 19 for J, given Jstar, set of j,k,Mstar,m1,m2, and fixed E.
    J00 and J01 are initial guesses for J.
    """
    k = sysprops[1]
    
    def Hamminusconstant(J,theta,Jstar,sysprops,E):
        return Ham_eqn19(J,theta,Jstar,sysprops) - E

    return optimize.root_scalar(Hamminusconstant,x0=J00,x1=J01,
                                args=(np.pi/k,Jstar0,sysprops,E)).root

def PR_P(P,sysprops):
    j,k,Mstar,m1,m2 = sysprops

    mu1 = mu(m1,Mstar)
    mu2 = mu(m2,Mstar)
    alpha00 = alpha0(j,k,Mstar,m1,m2)
    
    return (3*(mu1+mu2)/2*(j/mu2+(j-k)/(mu1*np.sqrt(alpha00))) * P +1)*j/(j-k)

def PR_min_max(sysprops,e1,e2,pomega1,pomega2,P1,P2):
    k = sysprops[1]
    Jsamp = J(sysprops,e1,e2,pomega1,pomega2)
    Jstar0 = Jstar(sysprops,e1,e2,pomega1,pomega2,P1,P2)

    JatEmax = max_Ham_eqn19(Jsamp,Jstar0,sysprops)
    Emax = Ham_eqn19(JatEmax,0,Jstar0,sysprops)
    Jmax = Ham_eqn19_Jsolver(JatEmax,Jsamp,Jstar0,sysprops,Emax)

    Pmax = (Jmax-Jstar0)/k
    Pmin = -Pmax

    PRmin = PR_P(Pmax,sysprops)
    PRmax = PR_P(Pmin,sysprops)

    return PRmin,PRmax

def PR_min_max_allin1(j,k,Mstar,m1,m2,P1,P2,e1,e2,pomega1,pomega2):    
    mu1 = m1*Mstar/(m1+Mstar)
    mu2 = m2*Mstar/(m2+Mstar)

    alpha0 = ((j-k)/j)**(2/3)*((Mstar+m1)/(Mstar+m2))**(1/3)

    A = 3*j*(mu1+mu2)/2*(j/mu2 + (j-k)/(mu1*np.sqrt(alpha0)))

    f,g = get_fg_coeffs(j,k)

    ftilde = np.sqrt((mu1+mu2)/(mu1*np.sqrt(alpha0)))*f
    gtilde = np.sqrt((mu1+mu2)/mu2)*g

    epsilon = m1*mu2/(Mstar*(mu1+mu2))
    epstilde = 2*(ftilde**2+gtilde**2)**(k/2)*epsilon

    def Ham(J,theta,Jstar):
        return -1/(2*k**2) *A *(J-Jstar)**2 - epstilde*J**(k/2)*np.cos(k*theta)

    #for our sample system
    Jsamp = (f**2*e1**2+g**2*e2**2+2*f*g*e1*e2*np.cos(pomega2-pomega1))/(
        ftilde**2+gtilde**2)
    Psamp = ((j-k)/j*(P2/P1) - 1 )/(
        3*(mu1+mu2)/2*(j/mu2+(j-k)/(mu1*np.sqrt(alpha0))))

    Jstar0 = Jsamp - k*Psamp

    #want to maximize Ham so minimize -Ham
    def oppHam(J,theta,Jstar):
        return -Ham(J,theta,Jstar)

    JatEmax = optimize.minimize(oppHam,Jsamp,args=(0,Jstar0))['x'][0]

    Emax = Ham(JatEmax,0,Jstar0)

    #want to solve for value so need to find roots of subtracting
    def Hamminusconstant(J,theta,Jstar,E):
        return Ham(J,theta,Jstar) - E

    deltaJ = Jsamp-JatEmax
    if abs(deltaJ)<1e-6:
        deltaJ = 1e-6
    Jmax = optimize.root_scalar(Hamminusconstant,x0=JatEmax,x1=JatEmax+deltaJ,args=(np.pi/k,Jstar0,Emax)).root
    Jmin = optimize.root_scalar(Hamminusconstant,x0=JatEmax,x1=JatEmax-deltaJ,args=(np.pi/k,Jstar0,Emax)).root

    Pmax_pos = (Jmax-Jstar0)/k
    Pmax_neg = (Jmin-Jstar0)/k

    PR1 = (3*(mu1+mu2)/2*(j/mu2+(j-k)/(mu1*np.sqrt(alpha0))) * Pmax_neg +1)*j/(j-k)
    PR2 = (3*(mu1+mu2)/2*(j/mu2+(j-k)/(mu1*np.sqrt(alpha0))) * Pmax_pos +1)*j/(j-k)

    if PR1<5/3 and PR2<5/3:
        PR2 = 5/3+(5/3-PR1)
    if PR1>5/3 and PR2>5/3:
        PR2 = 5/3-(PR1-5/3)

    return min(PR1,PR2),max(PR1,PR2)


##Mstar0 = 322776
##pomega10 = 0
##pomega20 = 0
##P10 = 30
##P20 = 5/3*1.001*P10
##m10 = 10
##m20 = 10
##
##sysprops0 = np.array([5,2,Mstar0,m10,m20])
##
##e1s = np.linspace(.01,.4,num=4)
##e2s = np.linspace(.01,.4,num=4)
##
##x1s = np.empty((len(e1s),len(e2s)))
##x2s = np.empty((len(e1s),len(e2s)))
##esplus = np.empty((len(e1s),len(e2s)))
##esdivide = np.empty((len(e1s),len(e2s)))
##for i,e1 in enumerate(e1s):
##    for l,e2 in enumerate(e2s):
##        esplus[i,l] = e1+e2
##        esdivide[i,l] = e1/e2
##        time0 = time.time()
##        x1s[i,l],x2s[i,l] = PR_min_max_allin1(5,2,Mstar0,m10,m20,P10,P20,
##                                       e1,e2,
##                                       pomega10,pomega20)
##        print(time.time()-time0)
####        x1s[i,l],x2s[i,l] = PR_min_max(sysprops0,e1,e2,0,0,P10,P20)
##
##plt.axvline(5/3,linestyle=':',color='k')
##plt.scatter(x1s,esplus,c=np.log10(esdivide))
##plt.scatter(x2s,esplus,c=np.log10(esdivide))
##
##plt.xlabel('period ratio')
##plt.ylabel('combined e')
##plt.colorbar(label='log e1/e2')
##plt.show()

