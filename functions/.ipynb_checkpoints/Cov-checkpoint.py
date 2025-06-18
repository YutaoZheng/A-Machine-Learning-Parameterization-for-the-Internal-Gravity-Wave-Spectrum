import numpy as np
from scipy.special import gamma
from gptide import cov
from scipy.special import kv as K_nu
import scipy

def matern_spectra(ff, params,lmbda=20):
    """General Matern PSD a la Lilly/Sykulski"""
    
    eta = params[0]
    alpha = params[1]
#     lmbda = params[2]

    c = (gamma(1/2) * gamma(alpha - 1/2)) / (2 * gamma(alpha) * np.pi)
    S = np.power(eta, 2) * np.power(lmbda, 2*alpha - 1) / c
    S *= np.power(np.power(2*np.pi*ff, 2) + np.power(lmbda, 2), -alpha)
    
    return S

# Calcualte the analytical result for the brodening peak spectrum
def LR_spectra(f,params, ℓ_cos1 = 0.5):
    eta,ℓ_exp1 = params
    ω_0 = 2*np.pi/ℓ_cos1
    ω = 2*np.pi*f
    c = l_exp1
    return np.power(eta,2.)/(np.power(ω-ω_0,2.)+np.power(c,2.))


#define acf
#From Sykulski 2015, equation 4 and 5
def LR_sykulski(dx,params,l_cos=0.5):
    
    eta = params[0]
    l_exp = params[1]
    
    ω_0 = 2*np.pi/l_cos
    LR_cos = np.exp(np.array([complex(0, imag) for imag in dx])*ω_0)
    c = 1/l_exp
    LR_exp = np.exp(-np.abs(dx)/l_exp)
    return np.power(eta,2.)/(2*c)*LR_cos*LR_exp

#define acf
def LR(dx,params,l_cos=2):
    '''
    l_cos: peak location (cpd)
    l_exp: peak broadening (day)
    '''    
    eta   = params[0]
    l_exp = params[1]   # day
    #convert unit
    ω_0 = 2*np.pi*l_cos   # radius
    c   = 1/l_exp   # radius
    
    LR_cos = np.cos(dx*ω_0)
    LR_exp = np.exp(-np.abs(dx)*c)
    
    return np.power(eta,2.)*LR_cos*LR_exp

def LR_2(dx,params,l_cos=2):
    '''
    l_cos: peak location (cpd)
    l_exp: peak broadening (day)
    '''    
    eta   = params[0]
    l_exp = params[1]   # day
    gamma = params[2]
    #convert unit
    ω_0 = 2*np.pi*l_cos   # radius
    c   = 1/l_exp   # r
    LR_cos = np.cos(dx*ω_0)
    LR_exp = np.exp(-np.power(np.abs(dx)*c,gamma))
    return np.power(eta,2.)*LR_cos*LR_exp

def LR_2_no_cos(dx,params,l_cos=2):
    '''
    l_cos: peak location (cpd)
    l_exp: peak broadening (day)
    '''    
    eta   = params[0]
    l_exp = params[1]   # day
    gamma = params[2]
    #convert unit
    c   = 1/l_exp   # r
    LR_exp = np.exp(-np.power(np.abs(dx)*c,gamma))
    return np.power(eta,2.)*LR_exp

# from speccy import acf
def Matern(dx, params,lmbda=0.5, sigma = 0, acf = True):
    """General Matern covariance a la Lilly/Sykulski"""

    eta = params[0]
    alpha = params[1]
#   lmbda = params[2]   # cpd

    lmbda = 2*np.pi*lmbda
    nu = alpha - 1/2
    K = 2 * np.power(eta, 2) / (gamma(nu) * np.power(2, nu))
    K *= np.power(np.abs(lmbda * dx), nu)
    with np.errstate(invalid='ignore'):
        K *= K_nu(nu, np.abs(lmbda * dx))
    K[np.isnan(K)] = np.power(eta, 2.)

    if acf:
        K[0] = K[0] + sigma**2
    else:
        n = dx.shape[0]
        K += sigma**2 * np.eye(n)
    
    return K

def M1L2(dx,params):
    
    O1_freq = 0.93 #cpd
    K1_freq = 1 #cpd
    S2_freq = 1.93 #cpd
    M2_freq = 2 #cpd
#     noise_var = params[0]
    η_matern1 = params[0]
    α_matern1 = params[1]
    eta1      = params[2]
    ℓ_exp1    = params[3]
    eta2      = params[4]
    ℓ_exp2    = params[5]
    
    matern1 = Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)              #background energy continuum  
    peak1 = LR(dx,(eta1,ℓ_exp1),l_cos=(O1_freq+K1_freq)/2)
    peak2 = LR(dx,(eta2,ℓ_exp2),l_cos=(S2_freq+M2_freq)/2)
    COV = matern1 + peak1 + peak2 
    return COV

def M1L1(dx,params):
    S2_freq = 1.93 #cpd
    M2_freq = 2 #cpd
    D2_freq = (S2_freq+M2_freq)/2
    
    η_matern1 = params[0]
    α_matern1 = params[1]
    eta_D2    = params[2]
    tau_D2    = params[3]
    gamma_D2  = params[4]
    
    matern1 = Matern(dx, (η_matern1,α_matern1),lmbda=3,sigma=1e-6)              #background energy continuum  
    peak2   = LR_2(dx, (eta_D2,tau_D2,gamma_D2),l_cos=D2_freq) 
    COV = matern1 + peak2 #+ noise
    return COV

def white_noise(dx,var):
    return var*scipy.signal.unit_impulse(len(dx)) #delta function

#when ω >> λ:
def Asymptote_Matern(ff, params,lmbda=3):
    eta = params[0]
    alpha = params[1]
    
    ff = 2*np.pi*ff
    lmbda = 2*np.pi*lmbda
    c = (gamma(1 / 2) * gamma(alpha - 1 / 2)) / (2 * gamma(alpha) * np.pi)
    S = np.power(eta, 2) * np.power(lmbda, 2 * alpha - 1) / c
    #when w >> λ:
    S *= np.power(np.power(ff, 2), -alpha)
    return S

#when ω >> ω_0:
#when ω >> ω_0+1/l_exp:
def Asymptote_LR(f, params):
    eta   = params[0]
    l_exp = params[1]   # day
    f = 2*np.pi*f
    c = 1/l_exp   # radius^-1
    return np.power(eta, 2.)*c*2/(f**2 )

