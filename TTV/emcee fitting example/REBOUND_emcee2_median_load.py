import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as c
from scipy import optimize,signal
import corner
import sys
import emcee
import mr_forecaster.mr_forecast as mr
from TTV.REBOUND_TTV import tns_tts_nplanet_l

filename = 'REB_emcee2.h5'
reader = emcee.backends.HDFBackend(filename)

params = ['m0','m1','m2',
          'P0','P1','P2',
          'l0','l1','l2',
          'esinom0','esinom1','esinom2',
          'ecosom0','ecosom1','escosom2']

samples = reader.get_chain()

steps,nwalkers,ndim = samples.shape
print(samples.shape)
tau = reader.get_autocorr_time(tol=0)
print(tau)
##sys.exit()

burnin = int(2*max(tau))

for i in range(ndim):
    plt.subplot(ndim+1,1,i+1)
    plt.axvline(x=burnin,color='gray')
    if i <= 2:
        plt.plot((samples[:,:,i]*u.Msun).to_value(u.Mearth),'k',alpha=.3)
    else:
        plt.plot(samples[:,:,i],'k',alpha=.3)
    plt.ylabel(params[i])
plt.show()

##tau = reader.get_autocorr_time()
##burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
##flat_samples = reader.get_chain(discard=burnin,flat=True)

for i in range(3):
    flat_samples[:,i] = (flat_samples[:,i]*u.Msun).to_value(u.Mearth)
    
print(np.median(flat_samples,axis=0))
print(np.percentile(flat_samples,15.9,axis=0))
print(np.percentile(flat_samples,84.1,axis=0))

cplot = corner.corner(flat_samples,labels=params,plot_contours=False,show_titles=True,
                      quantiles=[0.159, 0.5, 0.841])
plt.show()

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
masses = np.array([Mstar,0,0,0])
orbits = np.zeros((3,6))
for i in range(3):
    Mpli_posterior = mr.Rpost2M(Rpl_posterior[:,i],unit='Earth')
    Mpli = np.median(Mpli_posterior)*u.Mearth
    P = np.median(P_posterior[:,i])
    a = (((P*u.day)**2*c.G*(Mstar*u.Msun+Mpli)/(4*np.pi**2))**(1/3)).to_value(u.Rsun)
    b = np.median(b_posterior[:,i])
    inc = np.arccos(b*Rstar/a)
    orbits[i,2] = inc

#best fit model
MCMCfit = np.median(flat_samples,axis=0)

masses[1:] = (MCMCfit[:3]*u.Mearth).to_value(u.Msun)
orbits[:,0] = MCMCfit[3:6]
orbits[:,5] = MCMCfit[6:9]
esinoms = MCMCfit[9:12]
ecosoms = MCMCfit[12:]
orbits[:,1] = np.sqrt(esinoms**2+ecosoms**2)
orbits[:,4] = np.arctan2(esinoms,ecosoms)

pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
        t0,tmax,tref)

modeltts = np.empty(len(tts_all))
for i in range(3):
    ttsi = tts_mod[pflags_mod==i+1]
    tnsi = tns_mod[pflags_mod==i+1]
    ttsi_out = np.empty(len(tts_all[pflags_all==i+1]))
    for j,tn in enumerate(tns_all[pflags_all==i+1]):
        if tn in tnsi:
            ttsi_out[j] = ttsi[tnsi==tn]
    modeltts[pflags_all==i+1] = ttsi_out

sp1 = plt.subplot(221)
for i in range(3):
    Pm,Tm = np.polyfit(tns_all[pflags_all==i+1],modeltts[pflags_all==i+1],1)
    print(Pm,Tm)
    linear_tts_model = tns_all[pflags_all==i+1]*Pm + Tm
    TTVs_model = modeltts[pflags_all==i+1] - linear_tts_model
    plt.errorbar(tts_all[pflags_all==i+1],ttvs_all[pflags_all==i+1]*24*60,
                 xerr=tts_std_all[pflags_all==i+1],
                 yerr=ttvs_std_all[pflags_all==i+1]*24*60,fmt='.',color='C'+str(i))
    plt.plot(linear_tts_model,TTVs_model*24*60,'x',color='C'+str(i))
##plt.ylim(-100,100)
plt.title('median')
plt.subplot(223,sharex=sp1)
for i in range(3):
    Pm,Tm = np.polyfit(tns_all[pflags_all==i+1],modeltts[pflags_all==i+1],1)
    linear_tts_model = tns_all[pflags_all==i+1]*Pm + Tm
    TTVs_model = modeltts[pflags_all==i+1] - linear_tts_model
    plt.axhline(y=0,color='k')
    plt.errorbar(tts_all[pflags_all==i+1],(ttvs_all[pflags_all==i+1]-TTVs_model)*24*60,
                 xerr=tts_std_all[pflags_all==i+1],
                 yerr=ttvs_std_all[pflags_all==i+1]*24*60,fmt='.',color='C'+str(i))

#random model
plt.subplot(222,sharex=sp1,sharey=sp1)
ind_MCMC = np.random.randint(0,flat_samples.shape[0])

masses[1:] = (flat_samples[ind_MCMC,:3]*u.Mearth).to_value(u.Msun)
orbits[:,0] = flat_samples[ind_MCMC,3:6]
orbits[:,5] = flat_samples[ind_MCMC,6:9]
esinoms = flat_samples[ind_MCMC,9:12]
ecosoms = flat_samples[ind_MCMC,12:]
orbits[:,1] = np.sqrt(esinoms**2+ecosoms**2)
orbits[:,4] = np.arctan2(esinoms,ecosoms)

pflags_mod,tns_mod,tts_mod = tns_tts_nplanet_l(masses,orbits,
        t0,tmax,tref)

modeltts = np.empty(len(tts_all))
for i in range(3):
    ttsi = tts_mod[pflags_mod==i+1]
    tnsi = tns_mod[pflags_mod==i+1]
    ttsi_out = np.empty(len(tts_all[pflags_all==i+1]))
    for j,tn in enumerate(tns_all[pflags_all==i+1]):
        if tn in tnsi:
            ttsi_out[j] = ttsi[tnsi==tn]
    modeltts[pflags_all==i+1] = ttsi_out

for i in range(3):
    Pm,Tm = np.polyfit(tns_all[pflags_all==i+1],modeltts[pflags_all==i+1],1)
    print(Pm,Tm)
    linear_tts_model = tns_all[pflags_all==i+1]*Pm + Tm
    TTVs_model = modeltts[pflags_all==i+1] - linear_tts_model
    plt.errorbar(tts_all[pflags_all==i+1],ttvs_all[pflags_all==i+1]*24*60,
                 xerr=tts_std_all[pflags_all==i+1],
                 yerr=ttvs_std_all[pflags_all==i+1]*24*60,fmt='.',color='C'+str(i))
    plt.plot(linear_tts_model,TTVs_model*24*60,'x',color='C'+str(i))
##plt.ylim(-100,100)
plt.title('random')

plt.subplot(224,sharex=sp1)
for i in range(3):
    Pm,Tm = np.polyfit(tns_all[pflags_all==i+1],modeltts[pflags_all==i+1],1)
    linear_tts_model = tns_all[pflags_all==i+1]*Pm + Tm
    TTVs_model = modeltts[pflags_all==i+1] - linear_tts_model
    plt.axhline(y=0,color='k')
    plt.errorbar(tts_all[pflags_all==i+1],(ttvs_all[pflags_all==i+1]-TTVs_model)*24*60,
                 xerr=tts_std_all[pflags_all==i+1],
                 yerr=ttvs_std_all[pflags_all==i+1]*24*60,fmt='.',color='C'+str(i))


plt.show()
