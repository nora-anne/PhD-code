import matplotlib.pyplot as plt
import numpy as np
import rebound
import spock
from scipy import optimize
from TTV.REBOUND_TTV import tns_tts_nplanet_l
import mr_forecaster.mr_forecast as mr
from astropy import units as u
from astropy import constants as c
from Functions.orbits import PtoA
import argparse
import functools
print = functools.partial(print, flush=True)

def TTV_leastsquares(tt_data,masses,orbits,t0,tmax,tref,bounds):
    """
    tt_data is array of [tns,tts,pflags,tts_std]
    masses is array of masses [Mstar,mpl1,...] (solar mass)
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads. 
    t0/tmax are start/end of integration time (days).
    tref is the epoch for the orbital parameters.
    bounds is array of bounds for fitted parameters (shape [2,Npl*5])
    """

    Npl = len(masses)-1

    def rebound_pred(x,*ttv_params):
        """
        x is TTV data: transit numbers, transit times, and planet flags.
        ttv_params is an array of orbital parameters to be fit (m,P,l,esinom,ecosom for each planet).
        output is a set of predicted transit times.
        """
        ttv_params = np.array(ttv_params)
        tnum,ttime,pflag = x
        nonlocal tmax,t0

        eccs = np.sqrt(ttv_params[3::5]**2+ttv_params[4::5]**2)
        oms = np.arctan2(ttv_params[3::5],ttv_params[4::5])

        masses[1:] = ttv_params[0::5]
        orbits[:,0] = ttv_params[1::5]
        orbits[:,1] = eccs
        orbits[:,4] = oms
        orbits[:,5] = ttv_params[2::5]
        
        pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
                t0,tmax,tref)
        for i in range(Npl):
            if max(tns_mod[pflags_mod==i+1])<max(tnum[pflag==i+1]):
                tmax += orbits[i,0]
                t0 -= orbits[i,0]
                pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
                    t0,tmax,tref)

        transtimes_out = np.full(len(ttime),np.nan)
        for i in range(Npl):
            ttsi = tts_mod[pflags_mod==i+1]
            tnsi = tns_mod[pflags_mod==i+1]
            ttsi_out = np.empty(len(tts[pflags==i+1]))
            for j,tn in enumerate(tnum[pflag==i+1]):
                if tn in tnsi:
                    ttsi_out[j] = ttsi[tnsi==tn]
            transtimes_out[pflag==i+1] = ttsi_out

        return transtimes_out

    tns,tts,pflags,tts_std = tt_data

    p0 = []
    for i in range(Npl):
        p0.append(masses[i+1])
        p0.append(orbits[i,0])
        p0.append(orbits[i,5])
        p0.append(orbits[i,1]*np.sin(orbits[i,4]))
        p0.append(orbits[i,1]*np.cos(orbits[i,4]))
   
    fit,cov = optimize.curve_fit(rebound_pred,[tns,tts,pflags],
                                 tts, p0, sigma=tts_std,
                                 absolute_sigma=False,
                                 bounds = bounds)

    return fit,cov

def samples_make_physical(fit,cov,Nsamp=int(1e5)):
    """
    fit, cov are the outputs of the least squares fitting
    Nsamp is desired number of samples
    ensures that all masses are positive and e<1
    returns set of [Nsamp,Nparam] samples
    """

    samples = np.random.multivariate_normal(fit,cov,Nsamp)
    sample_es = np.sqrt(samples[:,3::5]**2+samples[:,4::5]**2)
    physical_ind = np.where((np.amax(sample_es,axis=1)<=1)&
                        (np.amin(samples[:,::5],axis=1)>0))[0]
    non_physical_ind = np.where((np.amax(sample_es,axis=1)>1)|
                        (np.amin(samples[:,::5],axis=1)<=0))[0]
    print('intially',len(physical_ind),'of',Nsamp,'physical')

    count = 0
    while len(physical_ind)<Nsamp and count<10*Nsamp:
        samples[non_physical_ind,:] = np.random.multivariate_normal(
            fit,cov,len(non_physical_ind))
        sample_es = np.sqrt(samples[:,3::5]**2+samples[:,4::5]**2)
        physical_ind = np.where((np.amax(sample_es,axis=1)<=1)&
                            (np.amin(samples[:,::5],axis=1)>0))[0]
        non_physical_ind = np.where((np.amax(sample_es,axis=1)>1)|
                            (np.amin(samples[:,::5],axis=1)<=0))[0]
        count += 1
        if count%(Nsamp)==0:
            print(count,len(physical_ind))
    if count==10*Nsamp:
        print('unable to find sufficient physical samples, returning',len(physical_ind))
        return samples[physical_ind,:]

    return samples

def samples_SPOCK(Mstar,incs,samples):
    """
    Mstar is stellar mass (solar mass)
    incs is fixed inclination of planets (rad)
    samples is [Nsamp,Nparam] samples
    returns SPOCK stability prediction for samples
    """

    Npl = len(incs)

    spock_stability = np.full(samples.shape[0],np.nan)
    for i,singlesample in enumerate(samples):
        mpls = singlesample[::5]
        Ps = singlesample[1::5]
        ls = singlesample[2::5]
        es = np.sqrt(singlesample[3::5]**2+singlesample[4::5]**2)
        oms = np.arctan2(singlesample[3::5],singlesample[4::5])
        
        model = spock.FeatureClassifier()
        sim = rebound.Simulation()
        sim.units = ['AU','day','Msun']
        sim.add(m=Mstar)
        for j in range(Npl):
            sim.add(m=mpls[j],
                    P=Ps[j],
                    l=ls[j],
                    e=es[j],
                    omega=oms[j],
                    Omega=0,
                    inc=incs[j])
        sim.move_to_com()
        spock_stability[i] = model.predict_stable(sim)
        if i%int(samples.shape[0]/10)==0:
            print(str(int(i/samples.shape[0]*100))+'%')
        
    return spock_stability

def samples_Zstable(Mstar,samples):
    """
    for 2 planet systems
    Mstar is stellar mass (solar mass)
    Assumes Omega=0
    samples is [Nsamp,10] samples
    returns HL18 Z and Z critical for samples (stable if Z<Z critical)
    """

    Z = np.full(samples.shape[0],np.nan)
    Z_critical = np.full(samples.shape[0],np.nan)
    for i,singlesample in enumerate(samples):
        mpls = singlesample[::5]
        Ps = singlesample[1::5]
        ls = singlesample[2::5]
        es = np.sqrt(singlesample[3::5]**2+singlesample[4::5]**2)
        oms = np.arctan2(singlesample[3::5],singlesample[4::5])
        Ms = np.concatenate(([Mstar],mpls))
        smas = (PtoA(Ms*u.Msun,Ps*u.day)).value
        ecross = (smas[1]-smas[0])/smas[0]
##        print((smas[1]-smas[0])/smas[0],(np.sum(mpls)/Mstar)**(2/7))

        Z[i] = abs(1/np.sqrt(2) * (es[1]*np.exp(oms[1]*1j)- es[0]*np.exp(oms[0]*1j)))
        Z_critical[i] = ecross/np.sqrt(2)*np.exp(
            -2.2*(np.sum(mpls)/Mstar)**(1/3)*(smas[1]/(smas[1]-smas[0]))**(4/3))
##        print(Z[i],Z_critical[i])

        if samples.shape[0]>1e3:
            if i%int(samples.shape[0]/10)==0:
                print(str(int(i/samples.shape[0]*100))+'%')
        
    return Z,Z_critical

def stable_physical_LS_fit(uid,Mstar,Rstar,Rpls,Ps,bs,tts,tts_std,tns,pflags,
                           stability_threshold=.05,Nsamp=int(1e5),
                           save_steps=True):
    """
    Mstar, Rstar in Msun,Rsun.
    Rpls is array of posteriors of planet radius (Earth radii).
    Ps is array of planet periods from posterior (days).
    bs is array of planet impact parameters from posterior.
    tts/tts_std is array of transit times/errors from posterior.
    tns is array of transit numbers for tts.
    pflags is array of planet flags for each transit.
    stability_threshold is the minimum allowable SPOCK prediction.
    Nsamp is the number of samples to return.
    If save_steps=True, will save numpy arrays of fit, cov, physical samples, and SPOCK predictions.
    returns physical, stable samples (Nsamp,5*Npl)
    """
    
    t0 = min(tts)-1
    tmax = max(tts)+1
    tref = .5*(t0+tmax)

    Npl = len(Ps)

    masses = np.array([Mstar])
    orbits = np.zeros((Npl,6))
    lower_bounds = []
    upper_bounds = []
    incs = np.empty(Npl)
    params = []

    p0 = []
    for i in range(Npl):
        Mpli_posterior = mr.Rpost2M(Rpls[:,i][Rpls[:,i]>=.1],unit='Earth')
        Mpli = np.median(Mpli_posterior)*u.Mearth
        masses = np.append(masses,Mpli.to_value(u.Msun))
        P = Ps[i]
        orbits[i,0] = P
        a = (((P*u.day)**2*c.G*(Mstar*u.Msun+Mpli)/(4*np.pi**2))**(1/3)).to_value(u.Rsun)
        b = bs[i]
        if np.isnan(b):
            inc = np.pi/2
        else:
            inc = np.arccos(b*Rstar/a)
        incs[i] = inc
        orbits[i,2] = inc
        try:
            mid_tn = int(np.median(tns[pflags==i+1]))
            if mid_tn in tns[pflags==i+1]:
                mid_tt = tts[pflags==i+1][tns[pflags==i+1]==mid_tn]
            else:
                mid_tt = [np.mean(tts)]
        except:
            mid_tt = [np.mean(tts)]
        l = ((tref-mid_tt)/P *2*np.pi)%(2*np.pi) - np.pi/2
        orbits[i,5] = l
        lower_bounds.append(0)
        lower_bounds.append(P-1)
        lower_bounds.append(-np.pi/2)
        lower_bounds.append(-.7)
        lower_bounds.append(-.7)
        ##            upper_bounds.append(.012) #~12.5 Mjup
        maxMpl = min(2*max(Mpli_posterior)*u.Mearth,(15*u.Mjup).to(u.Mearth))
        print('upper bound Mpl',maxMpl)
        upper_bounds.append(maxMpl.to_value(u.Msun)) #max from M-R relation
        upper_bounds.append(P+1)
        upper_bounds.append(2*np.pi - np.pi/2)
        upper_bounds.append(.7)
        upper_bounds.append(.7)
        params.append('m'+str(i+1))
        params.append('P'+str(i+1))
        params.append('l'+str(i+1))
        params.append('esinom'+str(i+1))
        params.append('ecosom'+str(i+1))
        p0.append(masses[i+1])
        p0.append(P)
        p0.append(l[0])
        p0.append(0)
        p0.append(0)

    bounds_p = [lower_bounds,upper_bounds]

    #check if p0 is stable (if not, use min mass not median)
    model = spock.FeatureClassifier()
    sim = rebound.Simulation()
    sim.units = ['AU','day','Msun']
    sim.add(m=Mstar)
    for i in range(Npl):
        sim.add(m=masses[1+i],
                P=Ps[i],
                l=orbits[i,5],
                e=0, 
                omega=0,
                Omega=0,
                inc=incs[i])
    sim.move_to_com()
    if Npl>2:
        spock_stability0 = model.predict_stable(sim)
        initial_stability = spock_stability0
    else:
        initial_stability = 1
        p = sim.particles
        ecross = (p[2].a-p[1].a)/p[1].a
        Z = abs(1/np.sqrt(2) * (.3*np.exp(p[2].omega*1j)-
                                .3*np.exp(p[1].omega*1j)))
        Z_critical = ecross/np.sqrt(2)*np.exp(
            -2.2*((p[1].m+p[2].m)/Mstar)**(1/3)*(
                p[2].a/(p[2].a-p[1].a))**(4/3))
        if Z/Z_critical > 1:
            initial_stability = 0
    if initial_stability==0:
        print('using minimum mass')
        masses = np.array([Mstar])
        orbits = np.zeros((Npl,6))
        lower_bounds = []
        upper_bounds = []
        incs = np.empty(Npl)
        params = []

        for i in range(Npl):
            Mpli_posterior = mr.Rpost2M(Rpls[:,i][Rpls[:,i]>=.1],unit='Earth')
            Mpli = min(Mpli_posterior)*u.Mearth
            masses = np.append(masses,Mpli.to_value(u.Msun))
            P = Ps[i]
            orbits[i,0] = P
            a = (((P*u.day)**2*c.G*(Mstar*u.Msun+Mpli)/(4*np.pi**2))**(1/3)).to_value(u.Rsun)
            b = bs[i]
            if np.isnan(b):
                inc = np.pi/2
            else:
                inc = np.arccos(b*Rstar/a)
            incs[i] = inc
            orbits[i,2] = inc
            try:
                mid_tn = int(np.median(tns[pflags==i+1]))
                if mid_tn in tns[pflags==i+1]:
                    mid_tt = tts[pflags==i+1][tns[pflags==i+1]==mid_tn]
                else:
                    mid_tt = [np.mean(tts)]
            except:
                mid_tt = [np.mean(tts)]
            l = ((tref-mid_tt)/P *2*np.pi)%(2*np.pi) - np.pi/2
            orbits[i,5] = l
            lower_bounds.append(0)
            lower_bounds.append(P-1)
            lower_bounds.append(-np.pi/2)
            lower_bounds.append(-.7)
            lower_bounds.append(-.7)
##            upper_bounds.append(.012) #~12.5 Mjup
            maxMpl = min(2*max(Mpli_posterior)*u.Mearth,(15*u.Mjup).to(u.Mearth))
            print('upper bound Mpl',maxMpl)
            upper_bounds.append(maxMpl.to_value(u.Msun)) #max from M-R relation
            upper_bounds.append(P+1)
            upper_bounds.append(2*np.pi - np.pi/2)
            upper_bounds.append(.7)
            upper_bounds.append(.7)
            params.append('m'+str(i+1))
            params.append('P'+str(i+1))
            params.append('l'+str(i+1))
            params.append('esinom'+str(i+1))
            params.append('ecosom'+str(i+1))

        bounds_p = [lower_bounds,upper_bounds]
    
    print('Starting LS fit')
    fit,cov = TTV_leastsquares([tns,tts,pflags,tts_std],
                           masses,orbits,t0,tmax,tref,bounds_p)
    print('LS fit complete, starting physical sample:',Nsamp)
    if save_steps:
        np.savez('LS_phys_stable_output_steps_'+uid,incs=incs,initmasses=masses,
                 fit=fit,cov=cov,p0=np.array(p0))

    samples = samples_make_physical(fit,cov,Nsamp)

    if Npl>2:
        print('physical samples complete, starting SPOCK analysis')
        if save_steps:
            np.savez('LS_phys_stable_output_steps_'+uid,incs=incs,initmasses=masses,
                     fit=fit,cov=cov,p0=np.array(p0),samples_phys=samples)

        spock_stability = samples_SPOCK(Mstar,incs,samples)
        print('SPOCK analysis complete')
        if save_steps:
            np.savez('LS_phys_stable_output_steps_'+uid,incs=incs,initmasses=masses,
                     fit=fit,cov=cov,p0=np.array(p0),samples_phys=samples,
                     spock_stability=spock_stability)

        stable_samp_ind = np.where(spock_stability>=stability_threshold)[0]
    else:
        print('physical samples complete, starting Z analysis')
        if save_steps:
            np.savez('LS_phys_stable_output_steps_'+uid,incs=incs,initmasses=masses,
                     fit=fit,cov=cov,p0=np.array(p0),samples_phys=samples)

        Z_samples,Zcrit_samples = samples_Zstable(Mstar,samples)
        Z_Zcrit = Z_samples/Zcrit_samples
        print('Z analysis complete')
        if save_steps:
            np.savez('LS_phys_stable_output_steps_'+uid,incs=incs,initmasses=masses,
                     fit=fit,cov=cov,p0=np.array(p0),samples_phys=samples,
                     Z_Zcrit=Z_Zcrit)

        stable_samp_ind = np.where(Z_Zcrit<1)[0]        

    Nstable = len(stable_samp_ind)
    print(Nstable,'stable samples \n')

    fit_stable = np.median(samples[stable_samp_ind,:],axis=0)
    lowerr_stable = fit_stable-np.percentile(samples[stable_samp_ind,:],15.9,axis=0)
    upperr_stable = np.percentile(samples[stable_samp_ind,:],84.1,axis=0)-fit_stable

    derived_eccs_samps = np.sqrt(samples[stable_samp_ind,3::5]**2+
                           samples[stable_samp_ind,4::5]**2)
    derived_oms_samps = np.arctan2(samples[stable_samp_ind,3::5],
                           samples[stable_samp_ind,4::5])
    derived_eccs = np.median(derived_eccs_samps,axis=0)
    derived_oms = np.median(derived_oms_samps,axis=0)
    derived_eccs_lowerr = derived_eccs-np.percentile(derived_eccs_samps,15.9,axis=0)
    derived_eccs_upperr = np.percentile(derived_eccs_samps,84.1,axis=0)-derived_eccs
    derived_oms_lowerr = derived_oms-np.percentile(derived_oms_samps,15.9,axis=0)
    derived_oms_upperr = np.percentile(derived_oms_samps,84.1,axis=0)-derived_oms

    derived_params_fit = np.concatenate((derived_eccs,derived_oms))
    derived_params_lowerr = np.concatenate((derived_eccs_lowerr,derived_oms_lowerr))
    derived_params_upperr = np.concatenate((derived_eccs_upperr,derived_oms_upperr))

    np.savez('LS_phys_stable_output_'+uid,fit=fit_stable,
             lower_error=lowerr_stable, upper_error=upperr_stable,
             derived_params_fit=derived_params_fit,
             derived_params_lower_error=derived_params_lowerr,
             derived_params_upper_error=derived_params_upperr)

    print('CALCULATED FIT')
    for i,p in enumerate(fit_stable):
        if i%5==0:
            print(params[i],(p*u.Msun).to_value(u.Mearth),'+',
                  (upperr_stable[i]*u.Msun).to_value(u.Mearth),'-',
                  (lowerr_stable[i]*u.Msun).to_value(u.Mearth))
        else:
            print(params[i],p,'+',upperr_stable[i],'-',lowerr_stable[i])
    for i,(e,slow,shigh) in enumerate(zip(derived_eccs,derived_eccs_lowerr,derived_eccs_upperr)):
        print('ecc'+str(i),e,'+',shigh,'-',slow)
    for i,(o,slow,shigh) in enumerate(zip(derived_oms,derived_oms_lowerr,derived_oms_upperr)):
        if o<0:
            print('omega'+str(i),o+2*np.pi,'+',shigh,'-',slow)
        else:
            print('omega'+str(i),o,'+',shigh,'-',slow)

    return samples[stable_samp_ind,:]

def chisq_TTV(p,ttv_data,masses,orbits,t0,tmax,tref):
    """
    p is the set of fitted parameters (m,P,l,esinom,ecosom for each planet)
    ttv_data is array of [tns,ttvs,pflags,ttvs_std]
    masses is array of masses [Mstar,mpl1,...] (solar mass)
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads. 
    t0/tmax are start/end of integration time (days).
    tref is the epoch for the orbital parameters.
    Calculates TTVs for model.
    Compares data TTVs and model TTVs using TTV error on both.
    Returns chi squared value and dof.
    """

    Npl = len(masses)-1

    def rebound_pred(x,*ttv_params):
        """
        x is TTV data: transit numbers and planet flags.
        ttv_params is an array of orbital parameters to be fit (m,P,l,esinom,ecosom for each planet).
        output is a set of predicted transit times.
        """
        ttv_params = np.array(ttv_params)
        tnum,pflag = x
        nonlocal tmax,t0

        eccs = np.sqrt(ttv_params[3::5]**2+ttv_params[4::5]**2)
        oms = np.arctan2(ttv_params[3::5],ttv_params[4::5])

        masses[1:] = ttv_params[0::5]
        orbits[:,0] = ttv_params[1::5]
        orbits[:,1] = eccs
        orbits[:,4] = oms
        orbits[:,5] = ttv_params[2::5]
        
        pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
                t0,tmax,tref)
        for i in range(Npl):
            if max(tns_mod[pflags_mod==i+1])<max(tnum[pflag==i+1]):
                tmax += orbits[i,0]
                t0 -= orbits[i,0]
                pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
                    t0,tmax,tref)

        transtimes_out = np.full(len(tnum),np.nan)
        for i in range(Npl):
            ttsi = tts_mod[pflags_mod==i+1]
            tnsi = tns_mod[pflags_mod==i+1]
            ttsi_out = np.empty(len(tts[pflags==i+1]))
            for j,tn in enumerate(tnum[pflag==i+1]):
                if tn in tnsi:
                    ttsi_out[j] = ttsi[tnsi==tn]
            transtimes_out[pflag==i+1] = ttsi_out
        
        return transtimes_out

    tns,ttvs,pflags,ttvs_std = ttv_data

    modeltts = rebound_pred([tns,pflags],*p)
    TTVs_model = np.array([])
    TTVs_model_std = np.array([])
    for i in range(Npl):
        if len(np.where(pflags_mod==i+1)[0])!=0:
            coeff,cov = np.polyfit(tns[pflags==i+1],
                                   modeltts[pflags==i+1],1,cov=True)
            Pm,Tm = coeff
            Pm_std,Tm_std = np.sqrt(np.diag(cov))
            linear_tts_model = tns[pflags==i+1]*Pm + Tm
            TTVs_modeli = modeltts[pflags==i+1] - linear_tts_model
            TTVs_modeli_std = np.sqrt((tns[pflags==i+1]*Pm_std)**2 + Tm_std**2)
            TTVs_model = np.concatenate((TTVs_model,TTVs_modeli))
            TTVs_model_std = np.concatenate((TTVs_model_std,TTVs_modeli_std))

    combined_std = np.sqrt(ttvs_std**2+TTVs_model_std**2)
    chisqv = np.sum(((ttvs-TTVs_model)/combined_std)**2)

    return chisqv, len(tns)-len(p)

def chisq_tt(p,tt_data,masses,orbits,t0,tmax,tref):
    """
    p is the set of fitted parameters (m,P,l,esinom,ecosom for each planet)
    tt_data is array of [tns,tts,pflags,tts_std]
    masses is array of masses [Mstar,mpl1,...] (solar mass)
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads. 
    t0/tmax are start/end of integration time (days).
    tref is the epoch for the orbital parameters.
    Compares data transit times and model transit times using error from data.
    Returns chi squared value and dof.
    """

    tns,tts,pflags,tts_std = tt_data

    eccs = np.sqrt(p[3::5]**2+p[4::5]**2)
    oms = np.arctan2(p[3::5],p[4::5])
    masses[1:] = p[0::5]
    orbits[:,0] = p[1::5]
    orbits[:,1] = eccs
    orbits[:,4] = oms
    orbits[:,5] = p[2::5]

    modeltts = output_tts(tt_data,masses,orbits,t0,tmax,tref)

    chisqv = np.sum(((tts-modeltts)/tts_std)**2)

    return chisqv, len(tns)-len(p)

def output_tts(tt_data,masses,orbits,t0,tmax,tref):
    """
    tt_data is array of [tns,tts,pflags,tts_std]
    masses is array of masses [Mstar,mpl1,...] (solar mass)
    orbits is an array of orbital arrays Keplerian orbital elements (P,e,inc,Om,om,l) in days/rads. 
    t0/tmax are start/end of integration time (days).
    tref is the epoch for the orbital parameters.
    Returns model transit times (matching to input ordering by transit numbers).
    """

    tns,tts,pflags,tts_std = tt_data

    Npl = len(masses)-1
    
    pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
            t0,tmax,tref)
    for i in range(Npl):
        if max(tns_mod[pflags_mod==i+1])<max(tns[pflags==i+1]):
            tmax += orbits[i,0]
            t0 -= orbits[i,0]
            pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
                t0,tmax,tref)

    transtimes_out = np.full(len(tts),np.nan)
    for i in range(Npl):
        ttsi = tts_mod[pflags_mod==i+1]
        tnsi = tns_mod[pflags_mod==i+1]
        ttsi_out = np.empty(len(tts[pflags==i+1]))
        for j,tn in enumerate(tns[pflags==i+1]):
            if tn in tnsi:
                ttsi_out[j] = ttsi[tnsi==tn]
        transtimes_out[pflags==i+1] = ttsi_out
    
    return transtimes_out
