import xarray as xr
import numpy as np
from . import bathmetry
from gptide import GPtideScipy

#this function samples the GP optimised above
def GP_IW_profile(single_point, Soln_table,time_length,covfunc,n_sample,phi_profile,mean_func,mean_params,var_name):
    """
    single_point- a given depth 
    soln_table: the solution table from the optimization
    time_length: the sample time length
    covfunc: the covariance function
    n_sample: the number of samples u wanna sample
    phi_profile: phi depth profile
    mean_func & mean_paras : the mean function & parameters found in harmonic analysis
    var_name: the name of the given variable (temp or velocity
    """
    #create a dataarry to contain the internal wave data varying with depth and time
    #n_sample represant how many sample drived from the GP process
    IW_profile = xr.DataArray(np.zeros((1,n_sample,len(time_length))), dims=('depth','sample','time'))
    #generate samples from the GP process
    ii = single_point
    noise = Soln_table.loc[ii][0]
    x = time_length[:,None]
    covparams = Soln_table.loc[ii][1:]
    GP1 = GPtideScipy(x, x, noise, covfunc, covparams, mean_func=mean_func,mean_params=mean_params.loc[ii][:])
    sample = GP1.prior(samples=n_sample).T
    IW_profile[0] = sample  #random samples
    
    #transfer to dataset and organising
    IW_profile= IW_profile.to_dataset(name=var_name)
    IW_profile.coords['depth']=('depth',[ii])
    IW_profile.coords['time'] = ('time',time_length)
    IW_profile.coords['sample'] = ('sample',np.arange(0,n_sample,1))  
    
    #create a dataarry to contain the internal wave data varying with depth and time
    size = np.zeros((1,n_sample,len(time_length)))
    IW_profile = IW_profile.assign(max_displacement=(['depth','sample','time',],np.zeros((1,n_sample,len(time_length)))))
    max_ζ_at_point = bathmetry.max_ζ_from_u_single_point(ii,IW_profile,phi_profile,var_name)  
    IW_profile.max_displacement[0] = max_ζ_at_point
    return IW_profile  #output the max displacement and the given var time profile

#this function interpolate the loose points
from scipy.interpolate import CubicSpline
def BC_interp(sample_profile,x_interp,var_name):
    """
    sample_profile: the max displacment time series sampled by GP
    x_interp: a time series that u want to interp for 
    var_name: the name of the variabale
    """
    x = sample_profile.time.values
    n_sample = sample_profile.sample.values
    sample_profile = sample_profile.assign_coords(time_2=x_interp)
    sample_profile = sample_profile.assign(var_interp=(['depth','sample','time_2',],np.zeros((1,len(n_sample),len(x_interp)))))  
    sample_profile = sample_profile.assign(displacement_interp=(['depth','sample','time_2',],np.zeros((1,len(n_sample),len(x_interp)))))
    
    for i in n_sample:
        y_var = sample_profile[var_name][0][i].values
        y_displacement = sample_profile.max_displacement[0][i].values
        cubic_interp_temp = CubicSpline(x, y_var)
        cubic_interp_displacment = CubicSpline(x, y_displacement)
        
        # Define points where you want to estimate values
        # Interpolate the values
        y_var_interp = cubic_interp_temp(x_interp)
        y_displacement_interp = cubic_interp_displacment(x_interp)
    
        sample_profile.var_interp[0][i] = y_var_interp
        sample_profile.displacement_interp[0][i] = y_displacement_interp
    return sample_profile


