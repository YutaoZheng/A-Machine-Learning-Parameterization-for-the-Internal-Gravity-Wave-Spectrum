import numpy as np
import xarray as xr
import scipy

#bathymetry
def Depth_tanh(beta, x):
    """
    Hyperbolic tangent shelf break

    H - total depth
    h0 - shelf height
    x0 - shelf break x location
    lt - shelf break width
    """
    
    H, h0, x0, lt = beta
    return H-0.5*h0*(1+np.tanh((x-x0)/(0.5*lt)))

#vertical density profile/stratification
def Single_tanh(z, params,):
    #return rho0 + rho1/2*(1-np.tanh( (z+z1)/h1))
    return params[0] - params[1]*np.tanh((z+params[2])/params[3])

#this function is based on the assumption of mode 1 structure and determines the vertcial shape of u
def Vertical_structure(phi_profile):
    """
    input: phi over depth
    output: the gradient of the phi profile, the depth of maximum displacement
    """
    z = phi_profile.depth.values
    dz = z[1]-z[0]
    grad = np.gradient(phi_profile.values,dz)
    structre = xr.DataArray(grad, coords={'depth': z}, dims=['depth'])
    max_depth = z[phi_profile.argmax()]
    return structre,max_depth

#phi is the vertical structure
#this function estimates the displacement from the given variable
def Max_ζ_from_u_single_point(single_point_depth,IW_depth_profile,phi_profile,var_name):
    """
    singple_point_depth - the given depth
    IW_depth_profile - the internal tides profile over depth
    phi_profile - phi depth profile
    var_name - the name of the given variable
    """
    j = single_point_depth #depth of a single point
    var_time_profile_at_j = IW_depth_profile[var_name].sel(depth=j,method='nearest')
    y,max_depth = u_vertical_structure(phi_profile)                 #find the scale for mode 1 velcotiy along the depth
    y_at_j = y.sel(depth=-j,method='nearest')
    a = var_time_profile_at_j/y_at_j
    y = a*y                                           #scale up the mode1 velcotiy along the depth to pass the single point
    Z = phi_profile.depth.values
    displacement = scipy.integrate.cumulative_trapezoid(y,Z,initial=0)#integrate y alond depth
    displacement = xr.DataArray(displacement,dims=['sample','time','depth',], coords={'sample':IW_depth_profile.sample ,'depth': Z,'time':IW_depth_profile.time},)
    displacement_max = displacement.sel(depth=max_depth,method='nearest') 
    return displacement_max #return the max displacement time profile at a depth 







