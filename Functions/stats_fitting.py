import numpy as np
from scipy import stats
import timeout_decorator
import warnings

distros = np.array(['alpha','anglit','arcsine','argus','beta','betaprime','bradford','burr'
,'burr12','cauchy','chi','chi2','cosine','crystalball','dgamma','dweibull'
,'erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife'
,'fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic'
,'gennorm','genpareto','genexpon','genextreme','gausshyper','gamma'
,'gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l'
,'halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant'
,'invgamma','invgauss','invweibull','johnsonsb','johnsonsu','kappa4'
,'kappa3','ksone','kstwobign','laplace','levy','levy_l','levy_stable'
,'logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke'
,'moyal','nakagami','ncx2','ncf','nct','norm','norminvgauss','pareto'
,'pearson3','powerlaw','powerlognorm','powernorm','rdist','rayleigh'
,'rice','recipinvgauss','semicircular','skewnorm','t','trapz','triang'
,'truncexpon','truncnorm','tukeylambda','uniform','vonmises'
,'vonmises_line','wald','weibull_min','weibull_max','wrapcauchy'])

def find_best_continuous_dist_warnings(data,printing=False):
    """
    Finds the 3 best continuous variable distributions to describe a data sample.

    data: dataset (1-D array)
    printing: if True, prints top 3 p-values from KS 2-sample test and corresponding distribution name.

    Displays all warnings.

    Returns the top 3 distribution names as strings.
    """

    @timeout_decorator.timeout(5)
    def fitsample(statobj,data):
        return statobj.fit(data)
    
    pvals = np.empty(len(distros))
    for i,distro in enumerate(distros):
        try:
            statobj = getattr(stats,distro)
            data_fit = fitsample(statobj,data)
            fit_sample = statobj.rvs(*data_fit,len(data))
            pvals[i] = stats.ks_2samp(data,fit_sample)[1]
        except:
            pvals[i] = 0
            if printing:
                print(distro,'failed')
                continue
        if printing:
            print(distro,pvals[i])

    top3_ind = np.argsort(pvals)[-3:]

    if printing:
        print('top 3 p-values',pvals[top3_ind])
        print('top 3 distros',distros[top3_ind])

    return distros[top3_ind]

def find_best_continuous_dist(data,printing=False):
    """
    Finds the 3 best continuous variable distributions to describe a data sample.

    data: dataset (1-D array)
    printing: if True, prints top 3 p-values from KS 2-sample test and corresponding distribution name.

    Returns the top 3 distribution names as strings.
    """
    warnings.simplefilter('ignore')
    return find_best_continuous_dist_warnings(data,printing)
    
