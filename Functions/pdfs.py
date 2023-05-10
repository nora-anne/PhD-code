import numpy as np

def double_Rayleigh_pdf(z,b):
    """
    pdf for the linear combination of 2 independent Rayleigh distros with same scale (b)
    """
    
    from scipy.special import erf
    return np.exp(-z**2/(2*b**2))/(4*b**3)*(2*b*z - np.exp(z**2/(4*b**2))*np.sqrt(np.pi)*
                        (2*b**2 - z**2)* erf(z/(2*b)))
