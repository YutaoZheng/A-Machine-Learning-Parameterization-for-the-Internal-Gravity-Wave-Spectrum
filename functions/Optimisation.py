import numpy as np
from speccy import utils as ut
from speccy import sick_tricks as gary
from scipy.optimize import minimize
from . import Processing

# import jax.numpy as jnp
# from jax.scipy.optimize import minimize
# from jax import jit

def whittle_function(params,n,delta,subset,I_ω,
                     covfunct):
    """
    Compute the Whittle likelihood function.

    Parameters:
        params (list or array): Model parameters.
        n: length of the time series
        delta: sampling internal
        index: index for selected frequency range
        I_ω: positive periodogoram
        covfunct: kernel for acf
        covfunct_spectra: kernel for spectra

    Returns:s
        float: Negative log-likelihood value.
    """
    tt = ut.taus(n, delta)
    I_ω_subset = I_ω[subset]
    #Nnumerical
    acf = covfunct(tt,params)                              #analytic acf
    ff,P_hat = gary.bochner(acf, delta=delta, bias = True) #numerical spectrum 
    P_hat = P_hat[ff>=0]   
    P_hat = P_hat[subset]
    whittle = -np.sum(np.log(P_hat) + I_ω_subset/P_hat)  
    return -whittle                                #negative whittle likelihood

# def whittle_fitting(params_ic,
#     method = 'Nelder-Mead'   #'L-BFGS-B'
#     myargs = (n,delta,subset,I_ω,covfunct)
#     def whittle_function_log(params_log, *args):
#         """
#         Wrapper for the Whittle likelihood function with log-transformed parameters.
#         """
#         # Transform parameters back to original scale.
#         params = np.exp(params_log)
#         return whittle_function(params, *args)

#     soln_whittle = minimize(
#         whittle_function_log,           # Whittle likelihood with log-transformed parameters.
#         x0=np.log(params_ic),           # Initial guess in log-space.
#         args=myargs,
#         method=method,
#         options={'maxiter': 10000})
    
#     best_soln_whittle = soln_whittle
#     best_funct = covfunct
#     params = np.exp(best_soln_whittle['x'])  # Transform best solution back to original scale.
#     # Numerical ACF and model spectrum.
#     tt = ut.taus(n, delta)
#     acf = covfunct(tt, params)
#     f_model, P_model = gary.bochner(acf, delta=delta, bias=True)
#     P_model = P_model[f_model >= 0] 
#     f_model = f_model[f_model >= 0]
    
#     return f_model[subset], P_model[subset], soln_whittle

# Logistic and inverse logistic transformations
def logistic(x, low, high):
    return low + (high - low) / (1 + np.exp(-x))    
def inverse_logistic(y, low, high):
    return np.log((y - low) / (high - y))

def transform_params(params_log, bounds=None, bound_idx=None):
    """Transform parameters to the appropriate space."""
    params = np.exp(params_log)
    if bound_idx is not None and bounds is not None:
        params[bound_idx] = logistic(params_log[bound_idx], *bounds)
    return params
def inverse_transform_params(params, bounds=None, bound_idx=None):
    """Inverse transform parameters back to the log space."""
    params_log = np.log(params)
    if bound_idx is not None and bounds is not None:
        params_log[bound_idx] = inverse_logistic(params[bound_idx], *bounds)
    return params_log

def whittle_fitting(params_ic, n, delta, subset, I_ω, covfunct, 
                    bound_idx=None, bounds=None):
    """
    Minimize the Whittle likelihood function with mixed transformations.
    """
    method = 'Nelder-Mead'
    myargs = (n, delta, subset, I_ω, covfunct)
    # Transform initial guess
    params_log_ic = inverse_transform_params(params_ic, bounds=bounds, bound_idx=bound_idx)

    # Wrapper for the Whittle likelihood
    def whittle_function_mixed(params_log, *args):
        params = transform_params(params_log.copy(), bounds=bounds, bound_idx=bound_idx)
        return whittle_function(params, *args)
    
    # Minimize the Whittle likelihood
    soln_whittle = minimize(
        whittle_function_mixed,
        x0=params_log_ic,
        args=myargs,
        method=method,
        options={'maxiter': 10000})

    # Transform best solution back to the original scale
    params = transform_params(soln_whittle['x'], bounds=bounds, bound_idx=bound_idx)
    # Numerical ACF and model spectrum
    tt = ut.taus(n, delta)
    acf = covfunct(tt, params)
    f_model, P_model = gary.bochner(acf, delta=delta, bias=True)
    P_model = P_model[f_model >= 0]
    f_model = f_model[f_model >= 0]

    return f_model[subset], P_model[subset], soln_whittle,


def Model_fit(P_ϵ_dict, time_dict, subset_dict,
              inital_guess,kernel, bound_idx = None,bounds=None):
    '''
    This function 
    input: obs periodogram list, obs time list, subset list,initial guess of the parameter, parameter bound, and kernel
    output: model fit frequency list, model fit periodogram list, parameter solution
    bound_idx (int, optional): Index of the parameter to constrain with bounds.
    '''
    F_model_fit_dict = {}
    P_model_fit_dict = {}
    Soln_model_fit_dict = {}
    Whittle_value_dict = {}
    
    for i in P_ϵ_dict:
        time = time_dict[i]
        subset = subset_dict[i]
        P_ϵ = P_ϵ_dict[i]

        time_length = len(time)
        delta_days = (time[1]-time[0]).astype('float')/1e9/86400
        f_model_fit, p_model_fit, \
        soln_model_fit = whittle_fitting(inital_guess,time_length,delta_days,                  
                                         subset,P_ϵ, kernel,
                                         bound_idx=bound_idx, bounds=bounds)
        F_model_fit_dict[i] = f_model_fit
        P_model_fit_dict[i] = p_model_fit
        if soln_model_fit['success']:
            Soln_model_fit_dict[i] = transform_params(soln_model_fit['x'], bounds=bounds, bound_idx=bound_idx)
            Whittle_value_dict[i]  = soln_model_fit['fun']  
        else:
            print('not converging at {} '.format(i))
            Soln_model_fit_dict[i] = np.full(len(soln_model_fit['x']),np.nan)
            Whittle_value_dict[i]  = soln_model_fit['fun']

    return F_model_fit_dict, P_model_fit_dict, Soln_model_fit_dict, Whittle_value_dict







