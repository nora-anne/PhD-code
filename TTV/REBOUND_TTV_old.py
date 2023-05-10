import matplotlib.pyplot as plt
import numpy as np
import rebound

def tns_tts_2planet(Mstar,m1,m2,orbit1,orbit2,t0,tmax):
    """
    Returns transit times and transit numbers for a 2 planet system.
    Uses the -z axis as the line of sight.
    Mstar, m1, m2 must have same units (assumed to be solar masses).
    orbit1/orbit2 are an array of Keplerian orbital elements (P,e,inc,Om,om,M) in days/rads.
    t0/tmax are start/end of integration time (days).
    """

    #set up simulation
    sim = rebound.Simulation()
    sim.units = ['Msun','AU','day']

    P1,e1,inc1,Om1,om1,M1 = orbit1
    P2,e2,inc2,Om2,om2,M2 = orbit2

    sim.t = t0
    
    sim.add(m=Mstar)
    sim.add(m=m1,P=P1,e=e1,inc=inc1,Omega=Om1,omega=om1,M=M1)
    sim.add(m=m2,P=P2,e=e2,inc=inc2,Omega=Om2,omega=om2,M=M2)
    sim.move_to_com()
    sim.integrator='mercurius'
    sim.dt = .02*P1

##    rebound.OrbitPlot(sim,slices=True)
##    plt.show()

    p = sim.particles

    #set up array for saving transit times and numbers (need numbers in case a transit is missed)
    transittimes1 = []
    transittimes2 = []
    transnums1 = []
    transnums2 = []
    i = 0
    k = 0
    i_offset = 0
    k_offset = 0
    tprogress = t0
    while sim.t <= tmax:
        x_old1 = p[1].x - p[0].x
        x_old2 = p[2].x - p[0].x
        t_old = sim.t
        sim.integrate(max(sim.t+0.5,tprogress)) # check for transits every 0.5 time units. Note that 0.5 is shorter than one orbit
        tprogress = sim.t
        t_new = sim.t
        x_new1 = p[1].x - p[0].x
        x_new2 = p[2].x - p[0].x
        transit1 = x_old1*x_new1<0 and p[1].z-p[0].z<0 
        transit2 = x_old2*x_new2<0 and p[2].z-p[0].z<0
        count = 0
        while transit1 and transit2: #both planets transiting in same step
##            print(sim.t)
            sim.integrate((t_new+t_old)/2)
            t_new = sim.t
            x_new1 = p[1].x - p[0].x
            x_new2 = p[2].x - p[0].x
            transit1 = x_old1*x_new1<0 and p[1].z-p[0].z<0 
            transit2 = x_old2*x_new2<0 and p[2].z-p[0].z<0
        if transit1 and ~transit2:
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old1*x_new1<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                x_new1 = p[1].x - p[0].x
                x_new2 = p[2].x - p[0].x
            transittimes1.append(sim.t)
            if i!=0 and transittimes1[-1] - transittimes1[-2] > 1.5*p[1].P and p[1].P>0: #time between transits indicates a missing transit
                deltatt = transittimes1[-1] - transittimes1[-2]
                deltatt_per1 = np.rint(deltatt/p[1].P)
                i_offset += deltatt_per1-1
            transnums1.append(i+i_offset)
            i += 1
            sim.integrate(sim.t+0.01)       # integrate to be past the transit
        if transit2 and ~transit1:
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old2*x_new2<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                x_new1 = p[1].x - p[0].x
                x_new2 = p[2].x - p[0].x
            transittimes2.append(sim.t)
            if k!=0 and transittimes2[-1] - transittimes2[-2] > 1.5*p[2].P and p[2].P>0:
                deltatt = transittimes2[-1] - transittimes2[-2]
                deltatt_per2 = np.rint(deltatt/p[2].P)
                k_offset += deltatt_per2-1
##                print(k,transittimes2[-1],transittimes2[-2],p[2].P,k_offset)
            transnums2.append(k+k_offset)
            k += 1
            sim.integrate(sim.t+0.01)       # integrateto be past the transit

##    rebound.OrbitPlot(sim,slices=True)
##    plt.show()
    
    return np.array(transnums1),np.array(transittimes1),np.array(transnums2),np.array(transittimes2)

def tns_tts_3planet(Mstar,m1,m2,m3,orbit1,orbit2,orbit3,t0,tmax):
    """
    Returns transit times and transit numbers for a 3 planet system.
    Uses the -z axis as the line of sight.
    Mstar, m1, m2, m3 must have same units (assumed to be solar masses).
    orbit1/orbit2/orbit3 are an array of Keplerian orbital elements (P,e,inc,Om,om,M) in days/rads.
    t0/tmax are start/end of integration time (days).
    """

    sim = rebound.Simulation()
    sim.units = ['Msun','AU','day']

    P1,e1,inc1,Om1,om1,M1 = orbit1
    P2,e2,inc2,Om2,om2,M2 = orbit2
    P3,e3,inc3,Om3,om3,M3 = orbit3

    sim.t = t0
    
    sim.add(m=Mstar)
    sim.add(m=m1,P=P1,e=e1,inc=inc1,Omega=Om1,omega=om1,M=M1)
    sim.add(m=m2,P=P2,e=e2,inc=inc2,Omega=Om2,omega=om2,M=M2)
    sim.add(m=m3,P=P3,e=e3,inc=inc3,Omega=Om3,omega=om3,M=M3)
    sim.move_to_com()
    sim.integrator='mercurius'
    sim.dt = .02*P1
    
    p = sim.particles

    #set up array for saving transit times and numbers (need numbers in case a transit is missed)
    transittimes1 = []
    transittimes2 = []
    transittimes3 = []
    transnums1 = []
    transnums2 = []
    transnums3 = []
    i = 0
    k = 0
    l = 0
    i_offset = 0
    k_offset = 0
    l_offset = 0
    tprogress = t0
    while sim.t <= tmax:
        x_old1 = p[1].x - p[0].x
        x_old2 = p[2].x - p[0].x
        x_old3 = p[3].x - p[0].x
        t_old = sim.t
        sim.integrate(max(sim.t+0.5,tprogress)) # check for transits every 0.5 time units. Note that 0.5 is shorter than one orbit
        tprogress = sim.t
        t_new = sim.t
        x_new1 = p[1].x - p[0].x
        x_new2 = p[2].x - p[0].x
        x_new3 = p[3].x - p[0].x
        transit1 = x_old1*x_new1<0 and p[1].z-p[0].z<0 
        transit2 = x_old2*x_new2<0 and p[2].z-p[0].z<0
        transit3 = x_old3*x_new3<0 and p[3].z-p[0].z<0
        while transit1 + transit2 + transit3 >1: #any 2 planets transiting in same step
            sim.integrate((t_new+t_old)/2)
            t_new = sim.t
            x_new1 = p[1].x - p[0].x
            x_new2 = p[2].x - p[0].x
            x_new3 = p[3].x - p[0].x
            transit1 = x_old1*x_new1<0 and p[1].z-p[0].z<0 
            transit2 = x_old2*x_new2<0 and p[2].z-p[0].z<0
            transit3 = x_old3*x_new3<0 and p[3].z-p[0].z<0
        if transit1:
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old1*x_new1<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                x_new1 = p[1].x - p[0].x
                x_new2 = p[2].x - p[0].x
                x_new3 = p[3].x - p[0].x
            transittimes1.append(sim.t)
            if i!=0 and transittimes1[-1] - transittimes1[-2] > 1.5*p[1].P and p[1].P>0: #time between transits indicates a missing transit
                deltatt = transittimes1[-1] - transittimes1[-2]
                deltatt_per1 = np.rint(deltatt/p[1].P)
                i_offset += deltatt_per1-1
            transnums1.append(i+i_offset)
            i += 1
            sim.integrate(sim.t+0.01)       # integrate to be past the transit
        if transit2:
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old2*x_new2<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                x_new1 = p[1].x - p[0].x
                x_new2 = p[2].x - p[0].x
                x_new3 = p[3].x - p[0].x
            transittimes2.append(sim.t)
            if k!=0 and transittimes2[-1] - transittimes2[-2] > 1.5*p[2].P and p[2].P>0:
                deltatt = transittimes2[-1] - transittimes2[-2]
                deltatt_per2 = np.rint(deltatt/p[2].P)
                k_offset += deltatt_per2-1
            transnums2.append(k+k_offset)
            k += 1
            sim.integrate(sim.t+0.01)       # integrateto be past the transit
        if transit3:
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old3*x_new3<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                x_new1 = p[1].x - p[0].x
                x_new2 = p[2].x - p[0].x
                x_new3 = p[3].x - p[0].x
            transittimes3.append(sim.t)
            if l!=0 and transittimes3[-1] - transittimes3[-2] > 1.5*p[3].P and p[3].P>0:
                deltatt = transittimes3[-1] - transittimes3[-2]
                deltatt_per3 = np.rint(deltatt/p[3].P)
                l_offset += deltatt_per3-1
            transnums3.append(l+l_offset)
            l += 1
            sim.integrate(sim.t+0.01)       # integrateto be past the transit

    return np.array(transnums1),np.array(transittimes1), \
           np.array(transnums2),np.array(transittimes2), \
           np.array(transnums3),np.array(transittimes3)

def tns_tts_nplanet(M,orbits,t0,tmax):
    """
    Returns transit times and transit numbers for an N planet system.
    Uses the -z axis as the line of sight.
    M is array of masses in same units (assumed to be solar masses).
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,M) in days/rads.
    t0/tmax are start/end of integration time (days).
    """

    N = len(M)-1
    Mstar = M[0]

    sim = rebound.Simulation()
    sim.units = ['Msun','AU','day']

    sim.t = t0
    
    sim.add(m=Mstar)
    for i in range(N):
        Pi,ei,inci,Omi,omi,Mi = orbits[i,:]
        sim.add(m=M[i+1],P=Pi,e=ei,inc=inci,Omega=Omi,omega=omi,M=Mi)
    sim.move_to_com()
    sim.integrator='mercurius'
    sim.dt = .02*orbits[0,0]
    p = sim.particles

##    rebound.OrbitPlot(sim,slices=True,xlim=(-1,1),ylim=(-1,1))
##    plt.show()

    #set up array for saving transit times and numbers (need numbers in case a transit is missed)
    transittimes = []
    transnums = []
    pflags = []
    lasttransittimes = np.zeros(N)
    index = np.zeros(N)
    index_offset = np.zeros(N)
    tprogress = t0
    while sim.t <= tmax:
        x_old = np.empty(N)
        for i in range(N):
            x_old[i] = p[i+1].x - p[0].x
        t_old = sim.t
        sim.integrate(max(sim.t+0.5,tprogress)) # check for transits every 0.5 time units. Note that 0.5 is shorter than one orbit
        tprogress = sim.t
        t_new = sim.t
        x_new = np.empty(N)
        for i in range(N):
            x_new[i] = p[i+1].x - p[0].x
        transit = np.full(N,False)
        for i in range(N):
            transit[i] = x_old[i]*x_new[i]<0 and p[i+1].z-p[0].z<0 
        while np.sum(transit)>1: #any 2 or more planets transiting in same step
            sim.integrate((t_new+t_old)/2)
            t_new = sim.t
            x_new = np.empty(N)
            for i in range(N):
                x_new[i] = p[i+1].x - p[0].x
                transit[i] = x_old[i]*x_new[i]<0 and p[i+1].z-p[0].z<0
        if np.sum(transit)==1:
            transit_index = np.where(transit==True)[0][0] #tells which index to use for transit
            planet_index = int(transit_index +1 ) #which planet is associated with that index
            ind = index[transit_index]
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old[transit_index]*x_new[transit_index]<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                for i in range(N):
                    x_new[i] = p[i+1].x - p[0].x
            transittimes.append(sim.t) #records transit time
            pflags.append(planet_index) #records which planet info is for
            if ind!=0:
                lasttransit = lasttransittimes[transit_index]
                deltatt = sim.t - lasttransit
                if deltatt > 1.5*p[planet_index].P and p[planet_index].P>0: #time between transits indicates a missing transit
                    deltatt_per = np.rint(deltatt/p[planet_index].P)
                    index_offset[transit_index] += deltatt_per-1
            transnums.append(ind+index_offset[transit_index])
            lasttransittimes[transit_index] = sim.t
            index[transit_index] += 1
            sim.integrate(sim.t+0.01)       # integrate to be past the transit

    return np.array(pflags),np.array(transnums),np.array(transittimes)

def tns_tts_nplanet_l_old(M,orbits,t0,tmax):
    """
    Returns transit times and transit numbers for an N planet system.
    Uses the -z axis as the line of sight.
    M is array of masses in same units (assumed to be solar masses).
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads.
    t0/tmax are start/end of integration time (days).
    """

    N = len(M)-1
    Mstar = M[0]

    sim = rebound.Simulation()
    sim.units = ['Msun','AU','day']

    sim.t = t0
    
    sim.add(m=Mstar)
    for i in range(N):
        Pi,ei,inci,Omi,omi,li = orbits[i,:]
        sim.add(m=M[i+1],P=Pi,e=ei,inc=inci,Omega=Omi,omega=omi,l=li)
    sim.move_to_com()
    sim.integrator='mercurius'
    sim.dt = .02*orbits[0,0]
    p = sim.particles

##    rebound.OrbitPlot(sim,slices=True,xlim=(-1,1),ylim=(-1,1))
##    plt.show()

    #set up array for saving transit times and numbers (need numbers in case a transit is missed)
    transittimes = []
    transnums = []
    pflags = []
    lasttransittimes = np.zeros(N)
    index = np.zeros(N)
    index_offset = np.zeros(N)
    tprogress = t0
    while sim.t <= tmax:
        x_old = np.empty(N)
        for i in range(N):
            x_old[i] = p[i+1].x - p[0].x
        t_old = sim.t
        sim.integrate(max(sim.t+0.5,tprogress)) # check for transits every 0.5 time units. Note that 0.5 is shorter than one orbit
        tprogress = sim.t
        t_new = sim.t
        x_new = np.empty(N)
        for i in range(N):
            x_new[i] = p[i+1].x - p[0].x
        transit = np.full(N,False)
        for i in range(N):
            transit[i] = x_old[i]*x_new[i]<0 and p[i+1].z-p[0].z<0 
        while np.sum(transit)>1: #any 2 or more planets transiting in same step
            sim.integrate((t_new+t_old)/2)
            t_new = sim.t
            x_new = np.empty(N)
            for i in range(N):
                x_new[i] = p[i+1].x - p[0].x
                transit[i] = x_old[i]*x_new[i]<0 and p[i+1].z-p[0].z<0
        if np.sum(transit)==1:
            transit_index = np.where(transit==True)[0][0] #tells which index to use for transit
            planet_index = int(transit_index +1 ) #which planet is associated with that index
            ind = index[transit_index]
            while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                if x_old[transit_index]*x_new[transit_index]<0.:
                    t_new = sim.t
                else:
                    t_old = sim.t
                sim.integrate( (t_new+t_old)/2.)
                for i in range(N):
                    x_new[i] = p[i+1].x - p[0].x
            transittimes.append(sim.t) #records transit time
            pflags.append(planet_index) #records which planet info is for
            if ind!=0:
                lasttransit = lasttransittimes[transit_index]
                deltatt = sim.t - lasttransit
                if deltatt > 1.5*p[planet_index].P and p[planet_index].P>0: #time between transits indicates a missing transit
                    deltatt_per = np.rint(deltatt/p[planet_index].P)
                    index_offset[transit_index] += deltatt_per-1
            transnums.append(ind+index_offset[transit_index])
            lasttransittimes[transit_index] = sim.t
            index[transit_index] += 1
            sim.integrate(sim.t+0.01)       # integrate to be past the transit

    return np.array(pflags),np.array(transnums),np.array(transittimes)

def tns_tts_nplanet_l(M,orbits,t0,tmax):
    """
    Returns transit times and transit numbers for an N planet system.
    Uses the -z axis as the line of sight.
    M is array of masses in same units (assumed to be solar masses).
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads.
    t0/tmax are start/end of integration time (days).
    """

    N = len(M)-1
    Mstar = M[0]

    sim0 = rebound.Simulation()
    sim0.units = ['Msun','AU','day']

    sim0.t = t0
    
    sim0.add(m=Mstar)
    for i in range(N):
        Pi,ei,inci,Omi,omi,li = orbits[i,:]
        sim0.add(m=M[i+1],P=Pi,e=ei,inc=inci,Omega=Omi,omega=omi,l=li)
    sim0.move_to_com()
    sim0.integrator='mercurius'
    sim0.dt = .02*orbits[0,0]
    p = sim0.particles
    sim0.integrate(t0-.1*orbits[0,0]) #start slightly before to make sure first transit not missed

##    rebound.OrbitPlot(sim,slices=True,xlim=(-1,1),ylim=(-1,1))
##    plt.show()

    #set up array for saving transit times and numbers (need numbers in case a transit is missed)
    transittimes = []
    transnums = []
    pflags = []
    for i in range(N):
        sim = sim0.copy()
        p = sim.particles
        lasttransittime = 0
        index = 0
        index_offset = 0
        tprogress = t0
        while sim.t <= tmax:
            x_old = p[i+1].x - p[0].x
            t_old = sim.t
            sim.integrate(max(sim.t+0.5,tprogress)) # check for transits every 0.5 time units. Note that 0.5 is shorter than one orbit
            tprogress = sim.t
            t_new = sim.t
            x_new = p[i+1].x - p[0].x
            if x_old*x_new<0 and p[i+1].z-p[0].z<0: 
                while t_new-t_old>1e-7:   # bisect until prec of 1e-7 reached
                    if x_old*x_new<0.:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    sim.integrate( (t_new+t_old)/2.)
                    x_new = p[i+1].x - p[0].x
                transittimes.append(sim.t) #records transit time
                pflags.append(i+1) #records which planet info is for
                if index!=0:
                    deltatt = sim.t - lasttransittime
                    if deltatt > 1.5*p[i+1].P and p[i+1].P>0: #time between transits indicates a missing transit
                        deltatt_per = np.rint(deltatt/p[i+1].P)
                        index_offset += deltatt_per-1
                transnums.append(index+index_offset)
                lasttransittime = sim.t
                index += 1
                sim.integrate(sim.t+0.01)       # integrate to be past the transit

    return np.array(pflags),np.array(transnums),np.array(transittimes)
