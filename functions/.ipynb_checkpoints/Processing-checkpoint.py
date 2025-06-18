#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
from s3fs import S3FileSystem, S3Map
from scipy import signal as sg
import pandas as pd

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
        
    return result[0]

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
0.

def Extract_good_data(raw_data,badidx):
    # Find the index of the last True in the beginning
    index_last_true_beginning = 0
    for i, value in enumerate(badidx):
        if value:
            index_last_true_beginning = i
        else:
            break
    # Find the index of the first True in the end
    index_first_true_end = len(badidx) - 1
    for i in range(len(badidx) - 1, -1, -1):
        if badidx[i]:
            index_first_true_end = i
        else:
            break
    # Select the good data (False values) between the last True in the beginning and the first True in the end
    good_data = raw_data[index_last_true_beginning + 1: index_first_true_end]
    return good_data

def Collect_temp(ncfiles):
    '''
    This function collects the time, depth, temperature, and the idx for bad data point from the ncfiles
    input:  ncfiles
    output: time_list,depths,temp_data_list,temp_badidx_list
    '''
    time_list = []
    temp_data_list = []
    temp_data_no_mean_list = []
    temp_badidx_list = []
    depths = [] #depth list
    
    fs = S3FileSystem(anon=True)
    
    for i in range(len(ncfiles)):
        ii = ncfiles[i]
        data = Open_file_nocache(ii,fs)
        
        obs_depth = data.instrument_nominal_depth
        depths.append(obs_depth)
    
        #get the raw temp time profile at this depth
        temp_raw,temp_badidx_raw = get_temp_qc_aodn(data, varname='TEMP')
        #remove the bad data occured at begin and end
        temp        = Extract_good_data(temp_raw,temp_badidx_raw)
        temp_badidx = Extract_good_data(temp_badidx_raw,temp_badidx_raw)
        #list append
        temp_data_list.append(temp.values)
        temp_badidx_list.append(temp_badidx.values)
        time_list.append(temp.TIME.values)

    return time_list,depths,temp_data_list,temp_badidx_list


#WINDOWING
def Find_window_idx(time_list,window_days):
    '''
    This function find the idx of the start and end point of the window in the time list
    input: time_list, window length in days
    output: idx of the window start and end point for time_list
    '''
    # data recording may not start at the same time
    window_idx_list = []
    for time in time_list:
        start = time[0]
        obs_duration = (time[-1] - start) #days
        n_window = int(obs_duration/np.timedelta64(window_days,'D'))
        if n_window < 1:
            print("duartion is smaller than window length")
        else:
            window_point = []
            window_idx = []
            for i in range(n_window+1):
                window_point.append(start+np.timedelta64(window_days,'D')*i)
            for ii in range(n_window):
                window_idx.append([np.logical_and(time>=window_point[ii] , time<=window_point[ii+1])])
        window_idx_list.append(window_idx) 
    return window_idx_list


def check_length(data_list):
    # Get the length of the first item
    first_item_length = len(data_list[0])

    # Assume all lengths are identical by default
    are_lengths_identical = True

    # Loop through the rest of the items and compare their lengths
    for item in data_list[1:]:
        if len(item) != first_item_length:
            are_lengths_identical = False
            break

    # Check if all lengths are identical
    if are_lengths_identical:
        return data_list
    else:
        print("Not all items in the list have the same length, check window_days")
        return data_list


def Windowing(data_list,window_idx_list):
    '''
    This function window the interested data list accourding to the provided window idx
    input: data list, window idx list
    output: windowed data list
    '''
    data_window_list = []
    for i,data in enumerate(data_list):
        window_idx = window_idx_list[i]
        
        data_window = []
        for idx in window_idx:
            data_window.append(data[idx[0]])
        
        data_window_list.append(data_window)
        
    return check_length(data_window_list)


## Find label list
def Find_window_label(windowed_time_period):
    """
    input: the time window list
    output: window start and end label list
            window year label list
            window only start label list
            window only end label list
    """
    year_list = []
    start_date_list = []
    end_date_list =[]
    
    for data in windowed_time_period:
        year       = []
        start_date = []
        end_date   = []
        
        window_length = len(data)
        for window in range(window_length):
            start_date.append(pd.to_datetime(data[window][0]).date())   #0-start
            end_date.append(pd.to_datetime(data[window][-1]).date())   #0-start
            year.append(pd.to_datetime(data[window][0]).year)  #0-start
            
        start_date_list.append(start_date)
        end_date_list.append(end_date)
        year_list.append(year)
        
    return year_list,start_date_list,end_date_list

def Find_mean_temp(windowed_time_period,windowed_temp_data,cutoff_freq,):
    
    filtered_temp_avg_list = []
    for order1,data in enumerate(windowed_temp_data):  #data at each depth
        filtered_temp_avg  = []
        for order2,data_window in enumerate(data):
            nyquist_freq  = 1/((windowed_time_period[order1][order2][1]-windowed_time_period[order1][order2][0]).astype('float')/1e9)*0.5  # Nyquist frequency in hz
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = sg.butter(4, normal_cutoff, btype='low', analog=False)
            filtered_temp = sg.filtfilt(b, a, data_window)
            filtered_temp_avg.append(np.mean(filtered_temp))
            
        filtered_temp_avg_list.append(filtered_temp_avg)
    return filtered_temp_avg_list

## Remove the mean of the whale data for the windowed list
def Remove_mean(data_windowed_list,badidx_windowed_list):
    data_with_mean_list = []
    data_without_mean_list = []

    for order1, data_windowed in enumerate(data_windowed_list):
        data_with_mean = []
        data_without_mean = []
        for order2, data in enumerate(data_windowed):
            #replace bad data with mean value
            data_replaced = np.where(badidx_windowed_list[order1][order2],data.mean(),data)
            #collected the replaced data with mean
            data_with_mean.append(data_replaced)
            #collected the replaced data without mean
            data_without_mean.append(data_replaced-np.mean(data_replaced))                         
        
        data_with_mean_list.append(data_with_mean)
        data_without_mean_list.append(data_without_mean)
        
    return data_with_mean_list,data_without_mean_list

def Cal_var(temp_data_list_window):  
    var_list = []
    for temp_data_window in temp_data_list_window:
        var = []
        for temp_data in temp_data_window:
            var.append(temp_data.var())
        var_list.append(var)
    return var_list

def Cal_var_from_Periodogram(f_list_window,p_list_window):  
    var_list = []
    for order1,f_list in enumerate(f_list_window):
        var = []
        for order2,f in enumerate(f_list):
            #find the freq bin
            freq_bin = f[1]-f[0]
            var.append(2*freq_bin*np.sum(p_list_window[order1][order2]))
        var_list.append(var)
    return var_list

def Cal_var_from_time(windowed_time_period,windowed_data,mean_func,cutoff_freq,):
    filtered_var_avg_list = []
    for order1,data in enumerate(windowed_data):  #data at each depth
        filtered_var_avg  = []
        for order2,data_window in enumerate(data):
            nyquist_freq  = 1/((windowed_time_period[order1][order2][1]-windowed_time_period[order1][order2][0]).astype('float')/1e9)*0.5  # Nyquist frequency in hz
            normal_cutoff = cutoff_freq / nyquist_freq
            b, a = sg.butter(4, normal_cutoff, btype='low', analog=False)
            
            filtered_var = sg.filtfilt(b, a, data_window-mean_func[order1][order2])
            filtered_var_avg.append(np.var(filtered_var))
            
        filtered_var_avg_list.append(filtered_var_avg)
    return filtered_var_avg_list
    


## PANDA DATAFRAME
# Constract the pd dataframe for results
def Transfer_list_to_df(data_list_period,depths,column_name):
    result_df = pd.DataFrame(data_list_period,index = depths)
    result_pd_series = pd.Series()
    for i in range(len(result_df.columns)):
        result_pd_series = result_pd_series.append(result_df[i])
    result_df = pd.DataFrame(result_pd_series,columns = [column_name])
    
    return result_df

#Add season
def Add_season(result_df):
    
    season_label_list = []# = ['Feb-Apr','May-Jul','Aug-Oct','Nov-Jan']
    for i in result_df['end_date']:
        month = i.month
        if   month in [3,4,5]:
            season = 'Feb-Apr'
        elif month in [6,7,8]:
            season = 'May-Jul'
        elif month in [9,10,11]:
            season = 'Aug-Oct'
        else: 
            season = 'Nov-Jan'
        
        season_label_list.append(season)
        
    result_df['season'] = season_label_list
        
    return result_df


def Create_df(start_date_list_period, 
              end_date_list_period, 
              year_list_period,
              temp_avg_list_period,
              total_var_list_period,
              HA_var_list_period,
              var_modulated_list_period,
              var_subtidal_list_period,
              depths_period):
    
    start_df     = Transfer_list_to_df(start_date_list_period, depths_period, 'start_date')
    end_df       = Transfer_list_to_df(end_date_list_period, depths_period, 'end_date')
    year_df      = Transfer_list_to_df(year_list_period, depths_period, 'year')
    mean_temp_df = Transfer_list_to_df(temp_avg_list_period, depths_period, 'mean_temp')
    total_var_df     = Transfer_list_to_df(total_var_list_period, depths_period, 'total_var')
    HA_var_df        = Transfer_list_to_df(HA_var_list_period, depths_period, 'HA_var')
    var_modulated_df = Transfer_list_to_df(var_modulated_list_period, depths_period, 'var_modulated')
    var_subtidal_df  = Transfer_list_to_df(var_subtidal_list_period, depths_period, 'var_subtidal')
   
    
    result_df_final = pd.concat([start_df, 
                                 end_df, 
                                 year_df, 
                                 mean_temp_df,
                                 total_var_df,
                                 HA_var_df,
                                 var_modulated_df,
                                 var_subtidal_df,],
                                axis=1)
    
    
    return Add_season(result_df_final)  #add season to the final df


# def Add_soln_to_df(soln_model_fit_df_final,parameter_name, whittile_df,final_df):
#     '''
#     input
#     final_df
#     '''
#     exploded_soln_df = soln_model_fit_df_final.explode('solution')
#     final_df_copy = final_df.copy()
#     for i in range(len(parameter_name)):
#         soln_value = exploded_soln_df.iloc[i::len(parameter_name)]
#         final_df_copy[parameter_name[i]] = soln_value
        
#     return pd.concat([final_df_copy, whittile_df],axis=1)  # combine the solution and whittle valu


def Finalise_df(soln_model_fit_df_final,
                soln_var_df_final,
                parameter_name, 
                whittile_df_final,
                site_name,
                model_type,
                final_df):
    '''
    input
    final_df
    '''
    #Add parameter solution to the result_df
    exploded_soln_df = soln_model_fit_df_final.explode('solution')
    final_df_copy = final_df.copy()
    for i in range(len(parameter_name)):
        soln_value = exploded_soln_df.iloc[i::len(parameter_name)]
        final_df_copy[parameter_name[i]] = soln_value
        
    #add whittle value to the result_df   
    final_df_copy = pd.concat([final_df_copy, whittile_df_final,soln_var_df_final],axis=1)  # combine the solution and whittle value
    
    #reset index
    final_df_copy.reset_index(inplace=True)
    final_df_copy.rename(columns={'index': 'depth'}, inplace=True)
    
    #add site in the first column
    final_df_copy.insert(0,'site',site_name)
    #add site in the first column
    final_df_copy.insert(7,'model_type',model_type)
    #sort the df by start_date and depth
    final_df_copy .sort_values(by=['start_date','depth'],inplace=True)
    final_df_copy['depth'] = -final_df_copy['depth']
    
    return final_df_copy.reset_index(drop=True)





#Sepectrum
#periodogarm
def Periodogram(data,sampling_interval):
    """
    input: time-series data
    output: periodogram
    """
    f_obs, Puu_obs = sg.periodogram(data, fs=1/sampling_interval,)
    return f_obs,Puu_obs/2 # devide by 2 !!

# def periodogram(time_data, sampling_interval = 1, h = None):
#     """
#     h:
#     """
#     n = time_data.size

#     if h is not None:
#         norm = np.sum(h**2)
#         scale = np.sqrt(n/norm)
#         time_data = scale * h * time_data

#     dft = np.fft.fft(time_data)/np.sqrt(n/delta)
    
#     I = np.real(dft * np.conj(dft))
#     ff = np.fft.fftfreq(n, delta)

#     return ut.fftshift(ff), ut.fftshift(I)


def Welch(data,sampling_interval,nperseg=2**15):
    """
    input: time-series data
    output: psd by welch
    """
    f_obs, Puu_obs = sg.welch(data, fs=1/sampling_interval,nperseg=nperseg,window='boxcar')
    return f_obs,Puu_obs/2 # devide by 2 !!


def Scale_periodogram(periodogram_list,badidx_list):
    """
    input: periodogram depth profile (in list), index of missing data over depth (in list)
    output: scaled periodogram (Parzen, 1996)
    """
    periodogram_list_copy = [row.copy() for row in periodogram_list] # Create a deep copy
    for i in range(len(badidx_list)):
        badidx = badidx_list[i]
        for ii in range(len(badidx)):
            r = np.sum(badidx[ii])/len(badidx[ii])  #r is the percent missing
            if r == 1:
                print ('100% missing data at depth idx of {} and removed during window {}'.format(i,ii))
            else:
                scale = 1/(1-r)
                periodogram_list_copy[i][ii] = periodogram_list_copy[i][ii]*scale

    return periodogram_list_copy #remove the all zero records which are from the shallow water





#SUBSET
def Select_frequency(data_list,index_list):
    selected_data_list = []
    for i in range(len(data_list)):
        data = data_list[i]
        index = index_list[i]
        selected_data = []
        for ii in range(len(data)):
            selected_data.append(data[ii][index[ii]])
        selected_data_list.append(selected_data)
    return selected_data_list

## select the frequency range from corilolis freq to buoyancy freq, and omit M4 frequency
def Subset_freq(ff_list, start_freq,end_freq, bandwidth, Omit=False, omit_freq=None):
    """
    Select the frequency range from Coriolis frequency to buoyancy frequency,
    and omit a specific frequency if Omit is True.

    Parameters:
    - ff_list: list of frequencies
    - start_freq: starting frequency 
    - end_freq: ending frequency 
    - Omit: boolean flag to indicate if omit_freq should be excluded (default is False)
    - omit_freq: frequency to be omitted if Omit is True

    Returns:
    - list of frequencies within the specified range, excluding omit_freq if Omit is True
    """
    if start_freq>=end_freq:
        raise ValueError("start_freq must be less than end_freq")
    
    subset_list = []
    for i in range(len(ff_list)):
        ff = ff_list[i]
        subset = []
        for ii in range(len(ff)):
            subset_range = np.less(start_freq + bandwidth,ff[ii]) & np.less(ff[ii],end_freq)
            # If Omit is True, remove omit_freq from the list
            if Omit:
                if omit_freq is None: 
                    raise ValueError("omit_freq must be provided when Omit is True")
                else: 
                    #omit
                    for freq in omit_freq:
                        omit_freq_range = np.less(freq-bandwidth, ff[ii]) & np.less(ff[ii], freq+bandwidth)
                    #AND its opposite
                        subset_range = subset_range & ~omit_freq_range
                    subset.append(subset_range)  
            else:
                subset.append(subset_range)  
                
        subset_list.append(subset)
    return subset_list

def Subset_peak(ff_list,peak_freq,peak_bandwidth):

    subset_list = []
    for i in range(len(ff_list)):
        ff = ff_list[i]
        subset = []
        for ii in range(len(ff)):
            subset_range = np.less(peak_freq - peak_bandwidth,ff[ii]) & np.less(ff[ii],peak_freq + peak_bandwidth)
            subset.append(subset_range)
        subset_list.append(subset)
    return subset_list

def Subset_semidiurnal_peak(ff_list):
    peak_bandwidth = 0.5
    M2_freq = 2 #cpd
    
    subset_list = []
    for i in range(len(ff_list)):
        ff = ff_list[i]
        subset = []
        for ii in range(len(ff)):
            subset_range = np.less(M2_freq - peak_bandwidth,ff[ii]) & np.less(ff[ii],M2_freq + peak_bandwidth)
            subset.append(subset_range)
        subset_list.append(subset)
    return subset_list


def Subset_residual(subset_diurnal,subset_semidiurnal,subset_internal_tide):
    subset = [[np.logical_and(np.logical_not(np.logical_or(a, b)),c) 
               for a, b,c in zip(diurnal, semidiurnal, internal_tide)]
               for diurnal, semidiurnal, internal_tide in zip(subset_diurnal,subset_semidiurnal,subset_internal_tide)]
    return subset


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























