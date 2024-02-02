from TTV.REBOUND_TTV import tns_tts_nplanet_l
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as c
from scipy import optimize
import emcee
import corner
import warnings
import mr_forecaster.mr_forecast as mr
warnings.filterwarnings('ignore',category=RuntimeWarning)

"""
Uses REBOUND to make models and fit from data with MCMC.
"""

#get transit times and calculate TTVs and std
tt_data = np.load('tt_data.npz')
tts_all = tt_data['tts_all']
tts_std_all = tt_data['tts_std_all']
tns_all = tt_data['tns_all']
pflags_all = tt_data['pflags_all']
ttvs_all = tt_data['ttvs_all']
ttvs_std_all = tt_data['ttvs_std_all']

t0 = min(tts_all)-1
tmax = max(tts_all)+1
tref = .5*(t0+tmax)

#get planet known planet properties, chosen from posterior
sys_param = np.load('sys_param.npz')
P_posterior = sys_param['P_posterior']
Mstar_posterior = sys_param['Mstar_posterior']
T0_posterior = sys_param['T0_posterior']
b_posterior = sys_param['b_posterior']
Rpl_posterior = (sys_param['Rpl_posterior'] *u.Rsun).to_value(u.Rearth)
Rstar_posterior = sys_param['Rstar_posterior']

Mstar = np.median(Mstar_posterior)
Rstar = np.median(Rstar_posterior)
masses = np.array([Mstar])
orbits = np.zeros((3,6))
for i in range(3):
    Mpli_posterior = mr.Rpost2M(Rpl_posterior[:,i],unit='Earth')
    Mpli = np.median(Mpli_posterior)*u.Mearth
    masses = np.append(masses,Mpli.to_value(u.Msun))
    P = np.median(P_posterior[:,i])
    orbits[i,0] = P
    a = (((P*u.day)**2*c.G*(Mstar*u.Msun+Mpli)/(4*np.pi**2))**(1/3)).to_value(u.Rsun)
    b = np.median(b_posterior[:,i])
    inc = np.arccos(b*Rstar/a)
    orbits[i,2] = inc
    mid_tn = int(np.median(tns_all[pflags_all==i+1]))
    mid_tt = tts_all[pflags_all==i+1][tns_all[pflags_all==i+1]==mid_tn]
    tt0 = tts_all[pflags_all==i+1][tns_all[pflags_all==i+1]==0]
    l = ((tref-mid_tt)/P *2*np.pi)%(2*np.pi) - np.pi/2
    orbits[i,5] = l
    orbits[i,4] = np.random.uniform(-np.pi,np.pi)

def rebound_pred(x,m0,m1,m2,P0,P1,P2,l0,l1,l2,esinom0,esinom1,esinom2,ecosom0,ecosom1,ecosom2):
    tnum,ttime,pflag = x

    ecc0 = np.sqrt(esinom0**2+ecosom0**2)
    ecc1 = np.sqrt(esinom1**2+ecosom1**2)
    ecc2 = np.sqrt(esinom2**2+ecosom2**2)

    om0 = np.arctan2(esinom0,ecosom0)
    om1 = np.arctan2(esinom1,ecosom1)
    om2 = np.arctan2(esinom2,ecosom2)

    masses[1:] = [m0,m1,m2]
    orbits[:,0] = [P0,P1,P2]
    orbits[:,1] = [ecc0,ecc1,ecc2]
    orbits[:,4] = [om0,om1,om2]
    orbits[:,5] = [l0,l1,l2]
    
    pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
            t0,tmax,tref)

    transtimes_out = np.full(len(ttime),np.nan)
    for i in range(3):
        ttsi = tts_mod[pflags_mod==i+1]
        tnsi = tns_mod[pflags_mod==i+1]
        ttsi_out = np.empty(len(ttime[pflag==i+1]))
        for j,tn in enumerate(tnum[pflag==i+1]):
            if tn in tnsi:
                ttsi_out[j] = ttsi[tnsi==tn]
        transtimes_out[pflag==i+1] = ttsi_out
    
    return transtimes_out

def ln_likelihood(p,pflag,tns_obs,tts_obs,ttvs_obs,ttvs_obs_std):
    modeltts = rebound_pred([tns_obs,tts_obs,pflag],*p)
    TTVs_model = np.array([])
    for i in range(3):
        Pm,Tm = np.polyfit(tns_obs[pflag==i+1],modeltts[pflag==i+1],1)
        linear_tts_model = tns_obs[pflag==i+1]*Pm + Tm
        TTVs_modeli = modeltts[pflag==i+1] - linear_tts_model
        TTVs_model = np.concatenate((TTVs_model,TTVs_modeli))
        
    lhood = -.5*np.sum((ttvs_obs-TTVs_model)**2/ttvs_obs_std**2 + np.log(ttvs_obs_std**2)) 
##    print(lhood)
    return lhood

def ln_prior(p):
    m0,m1,m2,P0,P1,P2,l0,l1,l2,esinom0,esinom1,esinom2,ecosom0,ecosom1,ecosom2 = p

    ecc0 = np.sqrt(esinom0**2+ecosom0**2)
    ecc1 = np.sqrt(esinom1**2+ecosom1**2)
    ecc2 = np.sqrt(esinom2**2+ecosom2**2)
    
    if (m0<=0 or m0>.012) or \
       (m1<=0 or m1>.012) or \
       (m2<=0 or m2>.012) or \
       (P0<=1.3 or P0>=5.5) or \
       (P1<=5.5 or P1>=11.3) or \
       (P2<=11.3 or P2>=18.5) or \
       (l0<-np.pi/2 or l0>2*np.pi) or \
       (l1<-np.pi/2 or l1>2*np.pi) or \
       (l2<-np.pi/2 or l2>2*np.pi) or \
       (ecc0<0 or ecc0>=1) or \
       (ecc1<0 or ecc1>=1) or \
       (ecc2<0 or ecc2>=1):
        #corresponds to 0-15 MJup, and then bound eccentricities
        return -np.inf
    return 0

def ln_probability(p,pflag,tn_obs,tt_obs,ttvs_obs,ttvs_obs_std):
    lp = ln_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    else:
        lhood = ln_likelihood(p,pflag,tn_obs,tt_obs,ttvs_obs,ttvs_obs_std)
        if np.isnan(lhood):
            return -np.inf
        prob = lp + lhood
        return prob

params = ['m0','m1','m2',
          'P0','P1','P2',
          'l0','l1','l2',
          'esinom0','esinom1','esinom2',
          'ecosom0','ecosom1','escosom2']

fit = np.load('curvefit_output2.npz')['values']
cov = np.load('curvefit_output2.npz')['cov']

ndim = len(params)
nwalkers = 32
p0 = np.random.multivariate_normal(fit,cov,nwalkers)
for j in range(nwalkers):
    while ~np.isfinite(ln_prior(p0[j,:])):
        p0[j,:] = np.random.multivariate_normal(fit,cov)

# Set up the backend
filename = 'REB_emcee2.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)    

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers,ndim,ln_probability,
                                args=(pflags_all,tns_all,tts_all,ttvs_all,ttvs_std_all),
                                backend=backend)

max_n = 100000
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

print('start')
# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n):
    # Only check convergence every 100 steps
    if (sampler.iteration % 100)==0:
        print(sampler.iteration)
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(params[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

fig.show()

flat_samples = sampler.get_chain(flat=True)
cplot = corner.corner(flat_samples,labels=params)
plt.show()
