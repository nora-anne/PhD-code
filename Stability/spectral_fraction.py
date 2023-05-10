import numpy as np
import rebound
import astropy.units as u

def sf_AMD_sim(sim,returneccs=False):
    """
    Takes rebound simulation as input.
    Integrates sim and calculates AMD over time.
    Returns spectral fraction for each planet (in sim index order).
    If returneccs, also returns the max ecc for each planet (in sma order).
    """

    p = sim.particles
    sim.integrator = 'WHFast'
    Ps = np.array([p[j].P for j in range(1,sim.N)])
    Pb = min(Ps)
    sim.dt = Pb/50 #50 timesteps in the inner orbit
    sim.collision = 'direct'

    N =  3000 #number of outputs
    t0 =  sim.t
    tmax = t0 + 5e6*Pb #run for 5 million orbits of inner planet
    times = np.linspace(t0,tmax,num=N)
    AMD = np.empty((sim.N-1,N))
    eccs_all = np.empty((sim.N-1,N))

    #integrate and calculate AMD for each planet and save output
    for i,time in enumerate(times):
        try:
            sim.integrate(time)
        except rebound.Collision as error:
            print('collision, time',sim.t)
            if returneccs==True:
                return np.full(3,np.nan),np.full(3,np.nan)
            else:
                return np.full(3,np.nan)

        #check for orbit crossing or unbound planet
        smasi = np.array([p[j].a for j in range(1,sim.N)])
        eccsi = np.array([p[j].e for j in range(1,sim.N)])
        if max(eccsi)>1:
            print('planet unbound, time',sim.t)
            if returneccs==True:
                return np.full(3,np.nan),np.full(3,np.nan)
            else:
                return np.full(3,np.nan)

        eccsi_sorted = eccsi[np.argsort(smasi)]
        eccs_all[:,i] = eccsi_sorted
        smasi_sorted = np.sort(smasi)
        orbseps = smasi_sorted[1:]*(1-eccsi_sorted[1:]) - (
            smasi_sorted[:-1]*(1+eccsi_sorted[:-1]))
        if min(orbseps) <= 0:
            print('orbit crossing, time',sim.t)
            if returneccs==True:
                return np.full(3,np.nan),np.full(3,np.nan)
            else:
                return np.full(3,np.nan)

        #calculate AMD for each planet
        for j in range(1,sim.N):
            AMD[j-1,i] = p[j].m*p[0].m/(p[j].m+p[0].m) * np.sqrt(
            sim.G*(p[j].m+p[0].m)*p[j].a)*(1-np.sqrt(1-p[j].e**2)*np.cos(p[j].inc))

    #find max eccs
    maxeccs = np.amax(eccs_all,axis=1)

    #use FFT to find the power spectra for each planet
    freqs = np.fft.rfftfreq(N,np.mean(np.diff(times)))
    AMD_spectral_fractions = np.empty(sim.N-1)
    for j in range(0,sim.N-1):
        pwr_spectra = abs(np.fft.rfft(AMD[j,:]))**2
        pwr_spectra_norm = pwr_spectra[freqs!=0]/max(pwr_spectra[freqs!=0])
        #spectral fraction is number of spikes above .05 per all freqs (except 0)
        AMD_spectral_fractions[j] = len(np.where(pwr_spectra_norm>=.05)[0])/len(freqs[freqs!=0])

    if returneccs==True:
        return AMD_spectral_fractions,maxeccs
    else:
        return AMD_spectral_fractions

def sf_AMD_simarch(simarch):
    """
    Takes rebound simulation archive as input.
    Returns spectral fraction for each planet.
    """

    N = len(simarch) #number of snapshots in SA

    #create arrays to hold data
    times = np.empty(N)
    AMD = np.empty((simarch[0].N-1,N))

    #get data from each instance in simarch
    for i,sim in enumerate(simarch):
        p = sim.particles
        times[i] = sim.t
        
        #check for orbit crossing
        smasi = np.array([p[j].a for j in range(1,sim.N)])
        eccsi = np.array([p[j].e for j in range(1,sim.N)])
        perisi = smasi*(1-eccsi)
        if min(perisi)<(3*u.Rsun).to_value(u.AU):
            print('collision with star, time',sim.t)
            return np.full(3,np.nan)
        if max(eccsi)>1:
            print('planet unbound, time',sim.t)
            return np.full(3,np.nan)
        eccsi_sorted = eccsi[np.argsort(smasi)]
        smasi_sorted = np.sort(smasi)
        orbseps = smasi_sorted[1:]*(1-eccsi_sorted[1:]) - (
            smasi_sorted[:-1]*(1+eccsi_sorted[:-1]))
        if min(orbseps) <= 0:
            print('orbit crossing, time',sim.t)
            return np.full(3,np.nan)

        #calculate AMD for each planet
        for j in range(1,sim.N):                   
            AMD[j-1,i] = p[j].m*p[0].m/(p[j].m+p[0].m) * np.sqrt(
                sim.G*(p[j].m+p[0].m)*p[j].a)*(1-np.sqrt(1-p[j].e**2)*np.cos(p[j].inc))

    #use FFT to find the power spectra for each planet
    freqs = np.fft.rfftfreq(N,np.mean(np.diff(times)))
    AMD_spectral_fractions = np.empty(sim.N-1)
    for j in range(0,sim.N-1):
        pwr_spectra = abs(np.fft.rfft(AMD[j,:]))**2
        pwr_spectra_norm = pwr_spectra[freqs!=0]/max(pwr_spectra[freqs!=0])
        #spectral fraction is number of spikes above .05 per all freqs
        AMD_spectral_fractions[j] = len(np.where(pwr_spectra_norm>=.05)[0])/len(freqs[freqs!=0])
    
    return AMD_spectral_fractions

def powerspectra_AMD_simarch(simarch):
    """
    Takes rebound simulation archive as input.
    Returns timestep, freqs, and power spectrum of AMD for each planet [Nsteps,Npl].
    """

    N = len(simarch) #number of snapshots in SA

    #create arrays to hold data
    times = np.empty(N)
    AMD = np.empty((simarch[0].N-1,N))

    #get data from each instance in simarch
    for i,sim in enumerate(simarch):
        p = sim.particles
        times[i] = sim.t

        #check for orbit crossing or ejection
        smasi = np.array([p[j].a for j in range(1,sim.N)])
        eccsi = np.array([p[j].e for j in range(1,sim.N)])
        perisi = smasi*(1-eccsi)
        if min(perisi)<(3*u.Rsun).to_value(u.AU):
            print('collision with star, time',sim.t)
            return np.full(3,np.nan)
        if max(eccsi)>1:
            print('planet unbound, time',sim.t)
            return np.full(3,np.nan)
        eccsi_sorted = eccsi[np.argsort(smasi)]
        smasi_sorted = np.sort(smasi)
        orbseps = smasi_sorted[1:]*(1-eccsi_sorted[1:]) - (
            smasi_sorted[:-1]*(1+eccsi_sorted[:-1]))
        if min(orbseps) <= 0:
            print('orbit crossing, time',sim.t)
            return np.full(3,np.nan)

        #calculate AMD for each planet
        for j in range(1,sim.N):
            AMD[j-1,i] = p[j].m*p[0].m/(p[j].m+p[0].m) * np.sqrt(
                sim.G*(p[j].m+p[0].m)*p[j].a)*(1-np.sqrt(1-p[j].e**2)*np.cos(p[j].inc))
                
    #use FFT to find the power spectra for each planet
    timestep = np.mean(np.diff(times))
    freqs = np.fft.rfftfreq(N,timestep) #cycles per unit of the sample spacing
    pwr_spectra = np.empty((len(freqs[freqs!=0]),sim.N-1))
    for j in range(0,sim.N-1):
        pwr_spectrum = abs(np.fft.rfft(AMD[j,:]))**2
        pwr_spectra[:,j] = pwr_spectrum[freqs!=0]/max(pwr_spectrum[freqs!=0])
        
    return timestep,freqs,pwr_spectra
