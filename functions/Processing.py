#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
from s3fs import S3FileSystem, S3Map
from scipy import signal as sg
import pandas as pd
import pickle
import gsw
from speccy import sick_tricks as gary
from speccy import ut
from datetime import datetime, timedelta
from . import Cov
from scipy.interpolate import interp1d
# Read the data from ncfile links

def Open_file_nocache(fname, myfs):
    """
    Load a netcdf file directly from an S3 bucket
    """
    fileobj = myfs.open(fname)
    return xr.open_dataset(fileobj)

def read_pickle(file_name):
    # Load data from the file
    with open('{}'.format(file_name), 'rb') as f:
        result = pickle.load(f)
    return result

def get_temp_qc_aodn(ds,varname='TEMP'):
    """
    Function that returns the QC's variable
    (only works with FV01 of the IMOS convention)
    """
    #find the index of bad quality data
    badidx1 = ds['{}_quality_control'.format(varname)].values <2
    #extra the data from the varnmae
    temp = ds['{}'.format(varname)]
    #replace the bad quality data with nan and find the index of them plus original nan
    badidx2 = np.isnan(temp.where(badidx1,np.nan))          
    return temp,badidx2   

def Extract_good_data(raw_data, badidx):
    # Convert to numpy arrays if not already
    badidx = np.asarray(badidx)
    # Find the first and last False indices (good data boundary)
    first_good_idx = np.argmax(~badidx)  # First False (good data) index
    last_good_idx = len(badidx) - np.argmax(~badidx[::-1]) - 1  # Last False index
    # Extract good data between the boundaries
    good_data = raw_data[first_good_idx:last_good_idx + 1]
    return good_data

def Collect_temp(ncfiles,site_name,order,window_size_days=80):
    '''
    This function collects the time, depth, temperature, and the idx for bad data point from the ncfiles
    input:  ncfiles
    output: time_list,depths,temp_data_list,temp_badidx_list
    '''
    ds_dict = {}
    ds_anomaly_dict = {}
    ds_time_dict = {}
    depth_dict   = {}
    fs = S3FileSystem(anon=True)
    for i in range(len(ncfiles)):
        ii = ncfiles[i]
        data = Open_file_nocache(ii,fs)
        #get the raw temp time profile at this depth
        temp_raw,temp_badidx_raw = get_temp_qc_aodn(data, varname='TEMP')
        #remove the bad data occured at begin and end
        temp        = Extract_good_data(temp_raw,temp_badidx_raw)
        temp_badidx = Extract_good_data(temp_badidx_raw,temp_badidx_raw)
        # #list append
        # temp_data_list.append(temp)
        # temp_badidx_list.append(temp_badidx.values)
        # time_list.append(temp.TIME)
        #windowing
        if window_size_days:
            temp_windowed_list,\
            temp_anomaly_windowed_list = windowing(temp, window_size_days=window_size_days)
            for window_order, temp_data_windowed in enumerate(temp_windowed_list):
                #put it into dict
                depth = temp_data_windowed.NOMINAL_DEPTH.values
                ds_name = '{}_P{}_{}_{}'.format(site_name, order,depth, window_order,)
                ds_dict[ds_name]          = temp_data_windowed
                depth_dict[ds_name]       = depth
                ds_time_dict[ds_name]     = temp_data_windowed['TIME'].values
                ds_anomaly_dict[ds_name]  = temp_anomaly_windowed_list[window_order]
                
    return depth_dict, ds_dict, ds_anomaly_dict, ds_time_dict

#WINDOWING
def windowing(ds,window_size_days=80):
    windowed_ds_list = []
    windowed_ds_anomaly_list = []
    time = ds.TIME
    obs_duration = (time[-1] - time[0]).values.astype('float')/1e9/86400 
    n_window     = int(obs_duration/window_size_days)
    if n_window < 1:
        print("duartion is smaller than window length")
    else:
        for i in range(n_window):
            start_time = time[0]+np.timedelta64(window_size_days,'D')*i
            end_time   = start_time+np.timedelta64(window_size_days,'D')
            if end_time > time[-1]:
                raise ValueError(f"Error: end_time {end_time.values} exceeds the dataset's last time value {time[-1].values}")
            else:
                windowed_ds = ds.sel(TIME=slice(start_time, end_time))
                windowed_ds_list.append(windowed_ds)
                #replace the nan with mean, then remove it
                windowed_ds[np.isnan(windowed_ds)] = windowed_ds.mean()
                windowed_ds_anomaly_list.append(windowed_ds-windowed_ds.mean())
    return windowed_ds_list,windowed_ds_anomaly_list

def Collect_time_info(time_dict):
    time_summary_dict = {}
    for key, time_array in time_dict.items():
        # Ensure the array isn't empty
        if len(time_array) > 0:
            # Extract the start and end date, and the year
            start_date = str(time_array[0])[:10]  # First date in the array
            end_date = str(time_array[-1])[:10]  # Last date in the array
            year = datetime.strptime(start_date, "%Y-%m-%d").year
            # Store the extracted information in the new dictionary
            time_summary_dict[key] = {'Year': year,
                                      'Start_date': start_date,
                                      'End_date': end_date}
    return time_summary_dict

def Find_mean_temp(time_dict,temp_dict,cutoff_freq=1/86400):
    temp_avg_dict = {}
    for i in temp_dict:
        time = time_dict[i]
        nyquist_freq  = 1/((time[1]-time[0]).astype('float')/1e9)*0.5  # Nyquist frequency in hz
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = sg.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_temp = sg.filtfilt(b, a, temp_dict[i].values)
        temp_avg_dict[i] = np.mean(filtered_temp)
    return temp_avg_dict

def Find_subtidal_signal(temp_dict,cutoff_freq):
    filtered_data_dict = {}
    for i in temp_dict:
        data = temp_dict[i]
        time = data.TIME
        nyquist_freq = 1/((time[1]-time[0]).astype('float')/1e9)*0.5
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = sg.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_data = sg.filtfilt(b, a, data)
        filtered_data_dict[i] = filtered_data
    return filtered_data_dict

def Calc_var_from_spectrum(f,p):
    freq_bin = f[1] - f[0]
    var      = 2*freq_bin*np.sum(p)
    return var


def Collect_var_info(temp_anomaly_dict, temp_anomaly_subtidal_dict, 
                     F_ε_subset_dict, P_ε_subset_dict):
    var_total_dict = {}
    var_subtidal_dict = {}
    var_ε_dict = {}
    for i in temp_anomaly_dict:
        var_total_dict[i]    = temp_anomaly_dict[i].var().values.item()
        var_subtidal_dict[i] = temp_anomaly_subtidal_dict[i].var()
        var_ε_dict[i]        = Calc_var_from_spectrum(F_ε_subset_dict[i],P_ε_subset_dict[i])

    return   var_total_dict,var_subtidal_dict,var_ε_dict
    
def Calc_Parameter_medium_and_std(df, selected_columns):
    # Select only numerical columns
    numeric_columns = df.select_dtypes(include='number').columns
    
    # Filter selected columns to include only numerical ones
    selected_columns = [col for col in selected_columns if col in numeric_columns]
    
    # Compute the median and standard deviation using 'Site' for grouping
    median_df = df.groupby('Site')[selected_columns].median().round(2)
    std_df    = df.groupby('Site')[selected_columns].std().round(2)

    # Merge the two dataframes on 'site' and format values as 'median (std)'
    df_final = median_df.copy()
    for col in selected_columns:
        df_final[col] = median_df[col].astype(str) + " (" + std_df[col].astype(str) + ")"
        
    return df_final


## PANDA DATAFRAME
# Constract the pd dataframe for results


#Add season
def Add_season(result_df): 
    season_label_list = []# = ['Feb-Apr','May-Jul','Aug-Oct','Nov-Jan']
    for i in result_df['Start_date']:
        month = pd.to_datetime(i).month
        if   month in [1,2,3]:
            season = 'Feb-Apr'
        elif month in [4,5,6]:
            season = 'May-Jul'
        elif month in [7,8,9]:
            season = 'Aug-Oct'
        else: 
            season = 'Nov-Jan'
        season_label_list.append(season)
    result_df['season'] = season_label_list
        
    return result_df

def Construct_soln_df_from_dict(time_summary_dict, var_summary_dict,avg_temp_dict,
                                  soln_model_fit_dict,whittle_value_dict,
                                  parameter_name):
    
    time_df           = pd.DataFrame.from_dict(time_summary_dict,orient='index')
    var_df            = pd.DataFrame.from_dict(var_summary_dict,orient='columns')
    avg_temp_df       = pd.DataFrame.from_dict(avg_temp_dict,orient='index',columns=['mean_temp'])
    whittle_list_df   = pd.DataFrame.from_dict(whittle_value_dict,orient='index',columns=['Whittle_value'])
    soln_model_fit_df = pd.DataFrame.from_dict(soln_model_fit_dict,orient='index',columns=parameter_name)
    # soln_model_fit_df = np.exp(soln_model_fit_df)
    #concatdf
    final_df_all = pd.concat([time_df,avg_temp_df,var_df,whittle_list_df,soln_model_fit_df],axis=1)
    #add site name and depth
    final_df_all.reset_index(drop=False, inplace=True)
    final_df_all['Site'] = final_df_all['index'].str.extract(r'([A-Za-z0-9]+)_')
    final_df_all['Depth'] = final_df_all['index'].str.extract(r'_(\d+\.\d+)_')              
    final_df_all = Add_season(final_df_all)
    final_df_all.sort_values(by=['Start_date','Depth'],inplace=True)
    final_df_all.rename(columns={"index": "Dict_name"}, inplace=True)
    return final_df_all.reset_index(drop=True)

#Data cleaning for final_df
def Find_γ_close_to_boundary(final_df,parameter_bounds, tolerance = 1.01):
    #only work for γ
    #Remove the rows where model parameters are too close to boundary
    index_list_close_to_boundary = []
    for idx,value in enumerate(final_df['γ_D2']):
        if value *tolerance >= parameter_bounds[1]:
            index_list_close_to_boundary.append(idx)
    return index_list_close_to_boundary

def integral_timescale(covfunc, covparams, delta = 0.01, N=1000):
    tt = ut.taus(N, delta)  # days
    acf = covfunc(tt, covparams)
    return 1./covparams[0]**2 * np.trapz(acf, tt)

def Find_index_significant_LR(df, cutoff_freq=10):
    # Initialize a list to store rows meeting the condition
    selected_rows = []
    for idx, row in df.iterrows():
        # Extract parameters for the current row
        params = row.loc['η_c':'γ_D2'].values
        # Analytically define the ACF
        n = 600000
        delta = 600 / 86400
        tt = ut.taus(n, delta)
        acf_true_Matern = Cov.Matern(tt, params, lmbda=3)
        acf_true_LR2 = Cov.LR_2(tt, params[2:], l_cos=1.965)

        # Numerically calculate the spectrum from ACF
        ff_LR2, S_bias_LR2 = gary.bochner(acf_true_LR2, delta, bias=True)
        ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)
        
        interpolate_LR2 = interp1d(ff_LR2, S_bias_LR2, bounds_error=False, fill_value="extrapolate")
        interpolate_Matern = interp1d(ff_Matern, S_bias_Matern, bounds_error=False, fill_value="extrapolate")

        S_LR2_at_cutoff = interpolate_LR2(cutoff_freq)
        S_Matern_at_cutoff = interpolate_Matern(cutoff_freq)

        # Check the condition
        if S_Matern_at_cutoff >= 2 * S_LR2_at_cutoff:
            pass
        else:
            selected_rows.append(idx)
    # Return the filtered DataFrame
    return df.loc[selected_rows]

#Sepectrum
#periodogarm

def Cal_periodogram(data,Δ):
    f,p = gary.periodogram(data,Δ)
    return f[f>=0],p[f>=0]

def Welch(data,sampling_interval,nperseg=2**15):
    """
    input: time-series data
    output: psd by welch
    """
    f_obs, Puu_obs = sg.welch(data, fs=1/sampling_interval,nperseg=nperseg,window='boxcar')
    return f_obs,Puu_obs/2 # devide by 2 !!

def Scale_periodogram(temp_dict,P_raw_dict):
    P_dict = {}
    for i in temp_dict:
        badidx = np.isnan(temp_dict[i])
        r = badidx.sum().values/len(badidx)
        if r == 1:
            print ('100% missing data at depth idx of {} and removed during window {}'.format(i,ii))
        else:
            scale = 1/(1-r)
            P_dict[i] = P_raw_dict[i]*scale
    return P_dict


#SUBSET
def Subset_freq(ff_dict,P_dict, start_freq,end_freq, bandwidth, Omit=False, omit_freq=None):
    if start_freq >= end_freq:
        raise ValueError("start_freq must be less than end_freq")
    if Omit and omit_freq is None:
        raise ValueError("omit_freq must be provided when Omit is True")
        
    subset_dict = {}
    F_ϵ_modulated_dict = {}
    Puu_ϵ_modulated_dict = {}
    for i in ff_dict:
        ff = ff_dict[i]
        data = P_dict[i]
        # Define the subset range
        subset_range = np.less(start_freq + bandwidth, ff) & np.less(ff, end_freq)
        # If Omit is True, remove specified frequencies
        if Omit:
            for freq in omit_freq:
                omit_freq_range = np.less(freq - bandwidth, ff) & np.less(ff, freq + bandwidth)
                subset_range = subset_range & ~omit_freq_range
        # Use the subset range to select data
        subset_dict[i] = subset_range
        F_ϵ_modulated_dict[i] = ff[subset_range]
        Puu_ϵ_modulated_dict[i] = data[subset_range]
    return subset_dict,F_ϵ_modulated_dict, Puu_ϵ_modulated_dict


#for find the coherent peak 
def Find_closest_index(data, target):
    min_difference = float('inf')  # Initialize with a large value
    closest_index = None

    for i, value in enumerate(data):
        difference = abs(value - target)
        if difference < min_difference:
            min_difference = difference
            closest_index = i

    return closest_index

def Coherent_peaks(peak_loc,amp_parameters,frequency):
    peaks = 0
    for i in range(len(peak_loc)):
        idx = Find_closest_index(frequency,peak_loc[i])
        amp = np.sqrt(np.power(amp_parameters[i],2)+np.power(amp_parameters[i+len(peak_loc)],2))
        peak = np.abs(amp)*sg.unit_impulse(len(frequency),idx)
        peaks = peaks + peak
    return peaks




#Startificationi
def Find_max_η_depth(data,lat,threshold):
    '''
    data: from Whole_Soln_df_M1L2_clean with selected site and season
    '''
    N2_list = []
    p_mid_list = []  
    CT_list = []
    p_list = []
    for year in data['Year'].unique():
#     for year in [2012]:    
            b = data[data['Year']==year].sort_values(by='depth_round').copy()
            p = gsw.conversions.p_from_z(b['depth_round'].unique(),lat)
            SA = np.array([34.6]*len(p))
            CT = gsw.conversions.CT_from_t(SA,b['mean_temp'],p)
            N2,p_mid = gsw.stability.Nsquared(SA,CT,p,lat)

            N2_list.append(N2)
            p_mid_list.append(p_mid)
            CT_list.append(CT)
            p_list.append(p)
         
    p_mid = np.concatenate(p_mid_list)
    N2 = np.concatenate(N2_list) 
    #df_N2
    data ={'N2':N2,'p_mid':p_mid}
    df = pd.DataFrame(data).sort_values(by='p_mid')
    n2_mean = df.groupby('p_mid')['N2'].mean().reset_index()
    #find the idx for max
    max_index = n2_mean['N2'].idxmax()
    max_value_depth_up   = gsw.conversions.z_from_p(n2_mean.loc[max_index, 'p_mid'],lat)+threshold
    max_value_depth_down = gsw.conversions.z_from_p(n2_mean.loc[max_index, 'p_mid'],lat)-threshold
        
    return max_value_depth_up, max_value_depth_down


def Find_avg_thermocline_depth(dataset,loc,lat,seasons,threshold=40):
    depth_list=[]
    for j, season in enumerate(seasons):
        data = dataset.loc[(dataset['Site'] =='{}'.format(loc)) & (dataset['season'] =='{}'.format(season))].copy()
        #find the depth of max η_c
        max_value_depth_up, max_value_depth_down = Find_max_η_depth(data,lat,threshold)
        thermocline_depth = (max_value_depth_up+max_value_depth_down)/2
        depth_list.append(thermocline_depth)
    
    return np.average(depth_list)

def Estimate_position(target_depth,depth_list):
    '''
    estimate the position of a specific depth in the depth list
    '''
    # Find the two depths in depth_list that are closest to the target depth
    depth_array = np.array(depth_list)
    if target_depth < depth_array.min() or target_depth > depth_array.max():
        raise ValueError(f"Target depth {target_depth} is out of the range of depth_list")
        
    # Find the indices of the two closest depths
    lower_index = np.searchsorted(depth_array[::-1], target_depth) - 1
    upper_index = lower_index + 1
    
    lower_depth = depth_array[lower_index]
    upper_depth = depth_array[upper_index]
    
    # Interpolate the position of the target depth
    interpolated_position = lower_index + (target_depth - lower_depth) / (upper_depth - lower_depth)
    
    return interpolated_position















