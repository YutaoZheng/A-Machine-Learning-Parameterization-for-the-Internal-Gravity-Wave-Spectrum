import numpy as np
import matplotlib.pyplot as plt
from speccy import utils as ut
from speccy import sick_tricks as gary
import seaborn as sns
from . import Cov
from . import Processing
import string

def Plot_fit_solution_over_depth(parameter_name,solution_list,model_name = 'M1L1'):
    
    num_subplots = len(parameter_name)
    rows = 3
    cols = int(np.ceil(num_subplots/rows))
    fig, axes = plt.subplots(rows, cols, figsize=(30, 15))
    axes = axes.flatten()
    fig.text(0.5, 0.9, '{} solution parameter'.format(model_name), ha='center', va='center')
    
    for order, period in enumerate(solution_list):
        for ii in range(len(parameter_name)):
            for iii in range(len(period['start_date'])):
                parameter_value = period.sel(depth = period['depth'],
                                             start_date = period['start_date'][iii],
                                             parameters = period['parameters'][ii])
                axes[ii].plot(parameter_value.solution,-parameter_value['depth'],label = parameter_value['start_date'].values,markersize=14)
                axes[ii].set_title(parameter_name[ii])
        
    handles, labels = [], []
    h, l = axes[0].get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)
    # Create a single legend
    fig.legend(handles, labels, loc=(0.8,0.05),fontsize='18')
    e
    return fig, axes 
    

def plot_model_fit(F_obs_ε_dict, P_obs_ϵ_dict,
                   F_gm, P_gm,
                   F_model_fit_dict, P_model_fit_dict):
    for i in F_model_fit_dict:
        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        # Add a shared subtitle
        fig.suptitle(f"{i}", fontsize=14, y=0.95)
        # Plot with x-axis log scale (First subplot)
        ax[0].plot(F_obs_ε_dict[i], P_obs_ϵ_dict[i], label='Residual Subset', alpha=0.5)
        ax[0].plot(F_model_fit_dict[i], P_model_fit_dict[i], label='Model Fit', linewidth=2)
        ax[0].plot(F_gm, P_gm,label='GM')
        ax[0].set_title("Log Scale", fontsize=12)
        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_xlim(0.5, 40)
        ax[0].set_ylim(1e-5, 10)
        ax[0].legend()
        # Plot without x-axis log scale (Second subplot)
        ax[1].plot(F_obs_ε_dict[i], P_obs_ϵ_dict[i], label='Residual Subset', alpha=0.5)
        ax[1].plot(F_model_fit_dict[i], P_model_fit_dict[i], label='Model Fit', linewidth=2)
        ax[1].set_title("Linear Scale", fontsize=12)
        ax[1].set_yscale("log")
        ax[1].set_xlim(1.5, 2.5)
        ax[1].set_ylim(1e-5, 10)
        
        
        # Adjust layout and show the figure
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust for suptitle
        plt.show()
        plt.close()


def Plot_indominating_cases(parameters):
    n = 600000
    delta = 600/86400
    tt = ut.taus(n, delta)
    #analyticly define the acf
    acf_true_M1L2 = Cov.M1L2(tt, parameters)
    acf_true_LR1 = Cov.LR(tt, parameters[2:],l_cos=1)
    acf_true_LR2 = Cov.LR(tt, parameters[4:],l_cos=2)
    acf_true_Matern = Cov.Matern(tt, parameters,lmbda=3)

    #numerically calculate spectrum from acf
    ff_M1L2, S_bias_M1L2 = gary.bochner(acf_true_M1L2, delta, bias=True)
    ff_LR1, S_bias_LR1 = gary.bochner(acf_true_LR1, delta, bias=True)
    ff_LR2, S_bias_LR2 = gary.bochner(acf_true_LR2, delta, bias=True)
    ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)

    S_M_Amp =  Cov.Asymptote_Matern(ff_M1L2, parameters, lmbda=3)
    S_L1_Amp = Cov.Asymptote_LR(ff_M1L2, parameters[2:])
    S_L2_Amp = Cov.Asymptote_LR(ff_M1L2, parameters[4:])
    
    plt.rcParams['font.size'] = 20
    plt.plot(ff_LR1[ff_LR1>=0], S_bias_LR1[ff_LR1>=0], label="LR1", linestyle="-.",color='r') 
    plt.plot(ff_LR2[ff_LR2>=0], S_bias_LR2[ff_LR2>=0], label="LR2", linestyle="-.",color='blue')  
    plt.plot(ff_Matern[ff_Matern>=0], S_bias_Matern[ff_Matern>=0], label="Matern", linestyle="-.",color='green')   
    plt.plot(ff_M1L2[ff_M1L2>=0], S_bias_M1L2[ff_M1L2>=0], label="M1L2",color='black')   
    
    plt.plot(ff_M1L2[ff_M1L2>=5],S_L1_Amp[ff_M1L2>=5],label='L1_asymptote',color='r') 
    plt.plot(ff_M1L2[ff_M1L2>=5],S_L2_Amp[ff_M1L2>=5],label='L2_asymptote',color='blue')
    plt.plot(ff_M1L2[ff_M1L2>=5],S_M_Amp[ff_M1L2>=5],label='M_asymptote',color='green') 
    plt.axvline(x=0.7, color='r', linestyle='--', alpha = 0.5)
    
    plt.axvline(x=1, color='g', linestyle='--', alpha = 0.5)
    plt.axvline(x=2, color='b', linestyle='--', alpha = 0.5)
    # plt.axvline(x=λ, color='r',linestyle='--', alpha = 0.5,label='λ')
    # plt.axvline(x=1+1/parameters[3],color='black', linestyle=':', alpha = 0.5,label='ω_1+c1')
    # plt.axvline(x=2+1/parameters[5],color='black', linestyle='-.', alpha = 0.5,label='ω_2+c2')
    # plt.title('Numerical Bochner')
    plt.text(0.7, 2, 'f', color='r', )
    plt.text(1, 2, 'D1', color='g', )
    plt.text(2, 2, 'D2', color='b', )
    plt.xlim(0.5,220)
    plt.ylim(1e-7, 5)
    plt.ylabel("PSD (K²/cpd)")
    plt.xlabel("Frequency (cpd)")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize = 20)
    plt.show()


def Plot_indominating_cases_for_M1P1(df,dict_name):
    #select the df
    selected_df = df[df['Dict_name'] == dict_name]
    params      = selected_df.loc[:, 'η_c':'γ_D2'].values[0]
    #analyticly define the acf
    n = 600000
    delta = 600/86400
    tt = ut.taus(n, delta)
    acf_true_M1L1   = Cov.M1L1(tt, params)
    acf_true_Matern = Cov.Matern(tt, params,lmbda=3)
    acf_true_LR2    = Cov.LR_2(tt, params[2:],l_cos=1.965)
    #numerically calculate spectrum from acf
    ff_M1L1, S_bias_M1L1 = gary.bochner(acf_true_M1L1, delta, bias=True)
    ff_LR2, S_bias_LR2 = gary.bochner(acf_true_LR2, delta, bias=True)
    ff_Matern, S_bias_Matern = gary.bochner(acf_true_Matern, delta, bias=True)
    #plot
    plt.plot(ff_LR2[ff_LR2>=0], S_bias_LR2[ff_LR2>=0], label="LR2", linestyle="-.",color='blue')  
    plt.plot(ff_Matern[ff_Matern>=0], S_bias_Matern[ff_Matern>=0], label="Matern", linestyle="-.",color='green')   
    plt.plot(ff_M1L1[ff_M1L1>=0], S_bias_M1L1[ff_M1L1>=0], label="M1P1",color='black') 
    plt.title(dict_name)
    plt.xlim(0.5,220)
    plt.ylabel("PSD (K²/cpd)")
    plt.xlabel("Frequency (cpd)")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize = 20)
    plt.show()  

def Plot_pair_plots(df, columns, colors, height=2.5, aspect=1):

    # Convert column names to LaTeX-style subscripts
    subscripted_columns = [r"$" + col.replace("_", r"_{") + "}$" if "_" in col else r"$" + col + "$" for col in columns]
    pair_plot_df = df[columns].copy()
    pair_plot_df.columns = subscripted_columns  # Rename columns for plotting
    # Create the pair plot
    g = sns.pairplot(pair_plot_df, diag_kind="kde", hue="${}$".format(columns[0]), palette=colors,
                     height=height, aspect=aspect)
                    #  markers=['o', 's', '^', 'v'], palette=colors)
    g.map_lower(sns.kdeplot, levels=4, color=".2")

    # Adjust legend marker size
    legend = g._legend
    legend.set_title("Site")  

    # Show the plot
    plt.show()
    return g
    

def Plot_depth_box_plot(locations,lat_list,data_df,bin_size,
                       parameter_names,parameter_units,colors,
                       save=False):
    
    # plot pre setup
    cols = 2
    rows = round(len(parameter_names)/cols)
    fig, axes = plt.subplots(rows, cols,sharey=True)
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_names))]
    axes = axes.flatten()
    # Plot box plots for each location
    # Set a thinner width
    width = 0.35 

    seasons = data_df['season'].drop_duplicates()
    for parameter_order,parameter_name in enumerate(parameter_names):
        for i, loc in enumerate(locations):
            #find the data at this location
            data = data_df.loc[data_df['Site'] ==loc].copy()
            depth_min = data['depth_round'].min()
            depth_max = data['depth_round'].max()
            depth_list = np.flip(np.arange(depth_min,depth_max+1,bin_size))  #from shallow to deep
             
            #depth_order is used to identify the location of yticks and box
            depth_order_list = []
            for depth_order, depth in enumerate(depth_list):
                #provide the position for ticks
                depth_order_list.append(depth_order)
                data_at_depth = data.loc[(data['depth_round'] >= (depth-bin_size)) 
                                         & (data['depth_round'] < (depth))][parameter_name].values  
                #select the non-nan values
                data_clean = [x for x in data_at_depth if not np.isnan(x)]
                #check data length
                if (len(data_clean)>=4):
                    axes[parameter_order].boxplot(data_clean,
                                                positions=[depth_order + i* width], 
                                                widths=width, 
                                                patch_artist=True,
                                                boxprops=dict(facecolor=colors[i]),
                                                medianprops=dict(color='black', linewidth=2),
                                                vert=False)
                  
            # Customize plot
            axes[parameter_order].set_title('({})'.format(figure_order[parameter_order]))
            parameter_name_subscript = r"$" + parameter_name.replace("_", r"_{") + "}$" if "_" in parameter_name else r"$" + parameter_name + "$"
            axes[parameter_order].set_xlabel('{} ({})'.format(parameter_name_subscript,parameter_units[parameter_order]))
            axes[parameter_order].set_yticks(depth_order_list[::3])  # Display depth
            axes[parameter_order].set_yticklabels(depth_list[::3])   # Set depth labels 
            axes[parameter_order].grid(True)
    #plot the thermocline depth
    for i,loc in enumerate(locations):
        avg_thermocline_depth = Processing.Find_avg_thermocline_depth(data_df,loc,lat_list[i],seasons,threshold=40)
        print(loc,avg_thermocline_depth)
        postision = Processing.Estimate_position(avg_thermocline_depth,depth_list)
        for j in [0,2]:
            axes[j].axhline(y=postision, color=colors[i], linestyle='--')
    
    #Create a custom legend for the locations
    location_legend = {location: color for location, color in zip(locations, colors)}
    handles = [plt.Rectangle((0,0),1,1, color=location_legend[location]) for location in locations]
    labels = locations   
    fig.legend(handles, labels, bbox_to_anchor=(0.6, 0.535))  
    for i in range(len(axes)):
        if i%2==0:
            axes[i].set_ylabel('Depth(m)')
    
    # Show plot
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rectangle to leave space for the y-axis label and legend
    plt.gca().invert_yaxis()
    if save:
        plt.savefig('{} depth profile of parameter.pdf'.format(locations))

    return fig, axes 

def Plot_seasonal_variability(df,parameter_name,parameter_units,lat_list,
                             colors,bbox_to_anchor,threshold = 20,save=False):
    #df_info
    locations = df['Site'].drop_duplicates()
    seasons = df['season'].drop_duplicates()
    figure_order = [list(string.ascii_lowercase)[i % 26] for i in range(len(parameter_name)*2)]
    #plot setup
    cols = 2
    rows = len(parameter_name)
    fig, axes = plt.subplots(rows, cols,sharey='row',sharex=True)
    axes = axes.flatten()
    positions = np.arange(len(locations))
    width = 0.1  # Set a thinner width
    #plotting
    for parameter_order,parameter_name in enumerate(parameter_name):
        for i, loc in enumerate(locations):
            for j, season in enumerate(seasons):
                data = df.loc[(df['Site'] =='{}'.format(loc)) & (df['season'] =='{}'.format(season))].copy()
                #find the depth of max η_c
                max_value_depth_up, max_value_depth_down = Processing.Find_max_η_depth(data,lat_list[i],threshold)
                #select the data at the depth of max η_c
                data_at_η_max = data.loc[(data['depth_round'] > max_value_depth_down)
                                        &(data['depth_round'] < max_value_depth_up)][parameter_name].values
                data_whole = data[parameter_name].values
                x_positions = np.full(len(data_at_η_max), i + j * width)
                #plot scatter for depth of max η_c
                axes[parameter_order*2].scatter(x_positions, data_at_η_max, color=colors[j])
                #plot boxplot for whole depth
                if len(data_whole)>4:
                    axes[parameter_order*2+1].boxplot(data_whole,
                                positions=[i + j * width], 
                                widths=width, patch_artist=True,
                                boxprops=dict(facecolor=colors[j]),
                                medianprops=dict(color='black', linewidth=2))   
        parameter_name_subscript =r"$" + parameter_name.replace("_", r"_{").replace("{", r"{") + "}$"           
        axes[parameter_order*2].set_ylabel('{} ({})'.format(parameter_name_subscript ,parameter_units[parameter_order]),
                                           fontsize=14)
        
    # Create a custom legend for the seasons
    season_legend = {season: color for season, color in zip(seasons, colors)}
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=season_legend[season], markersize=10) for season in seasons]
    labels = seasons
    fig.legend(handles, labels, title='Season', loc=(0.8, 0.5), \
           bbox_to_anchor=bbox_to_anchor, fontsize='medium',  handletextpad=0.25)

    for i in range(len(axes)):
        axe = axes[i]
        axe.set_title(f'({figure_order[i]})')
        axe.set_xticks(positions + (len(seasons) - 1) * width / 2)
        axe.set_xticklabels(locations)
        axe.grid(True)
    
    axes[0].set_title('At Thermocline Depth \n (a)')
    axes[1].set_title('Whole Depth \n (b) ')
    
    # Show plot
    fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rectangle to leave space for the y-axis label and legend
    # Show plot
    if save:
        plt.savefig('Seasonal_variability_of_parameter_at_max_eta_and_whole_depth.pdf', bbox_inches='tight')
    #plt.show() 

    return fig, axes 

def Plot_HA_result(time_list,A_list,xcoords,
                   Mean_params_list,ϵ_list,Yd_mean_list,
                   F_ϵ_list,Puu_ϵ_list, ):
    
    x = time_list
    y = A_list
    ϵ = ϵ_list
    yd_mean = Yd_mean_list
    #calculate spectrum
    Δ = (x[1]-x[0]).astype('float')/1e9/86400
    F_obs,P_obs = Processing.Cal_periodogram(y.values,Δ)

    plt.subplot(2, 1, 1)
    idx = 15000
    plt.plot(x[10000:idx],y[10000:idx],label='Obs',alpha=0.5)
    plt.plot(x[10000:idx],ϵ[10000:idx],'-.',label = 'Residual')
    plt.plot(x[10000:idx],yd_mean[10000:idx],label='HA',linewidth=2.5)
    plt.xlabel('days')
    plt.ylabel('Amp (m)')
    # plt.grid(b=True,ls=':')
    plt.title('Time series')
    plt.legend(loc="lower right")

    F_ϵ   = F_ϵ_list
    Puu_ϵ = Puu_ϵ_list
    Peaks = Processing.Coherent_peaks(xcoords[1:],Mean_params_list,F_ϵ)

    plt.subplot(2, 1, 2)
    plt.plot(F_obs,P_obs,label='Obs',alpha=0.5)
    plt.plot(F_ϵ,Puu_ϵ,label='Residual')
    plt.plot(F_ϵ,Peaks,label='HA',linewidth=2.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('f_mean[cycles per day]')
    plt.ylabel('PSD(m²/cpd)')
    # plt.grid(b=True,ls=':')
    plt.legend(loc="lower right") 
    plt.ylim(1e-5, 1e3)
    plt.xlim(0.4,50)


def Plot_obs_subplots(time_list, A_list, Mean_params_list, F_GM, P_GM):
    # Calculate spectrum from the time series
    x = time_list
    y = A_list
    Δ = (x[1] - x[0]).astype('float') / 1e9 / 86400
    F_obs, P_obs = Processing.Cal_periodogram(y.values, Δ)
    # Peaks = Processing.Coherent_peaks(xcoords[1:], Mean_params_list, F_obs)
    # Define frequency values for vertical lines
    O1_freq = 0.93  
    K1_freq = 1
    M2_freq = 1.93
    S2_freq = 2
    lat = A_list.LATITUDE.values
    f_coriolis = 4 * np.pi / 86400 * np.sin(lat * np.pi / 180)
    f_coriolis_cpd = np.abs(f_coriolis * 86400 / (2 * np.pi))
    xcoords = [O1_freq, K1_freq, M2_freq, S2_freq]
    Peaks = Processing.Coherent_peaks(xcoords, Mean_params_list, F_obs)
    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(2, 1,)
    ax1, ax2 = axes

    # Left subplot: Original plot in log-log scale
    ax1.plot(F_obs, P_obs, label='Observation', alpha=0.5)
    ax1.plot(F_obs, Peaks, label='Harmonic Analysis')
    ax1.plot(F_GM, P_GM, label='Garret-Munk', linewidth=2.5)
    ax1.axvline(x=O1_freq, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=K1_freq, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=M2_freq, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=S2_freq, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=f_coriolis_cpd, color='black', linestyle='--', alpha=0.5)
    ax1.text(f_coriolis_cpd - 0.05, 0.9, 'f', color='black')
    ax1.text(O1_freq - 0.25, 0.9, 'O1  K1', color='black')
    ax1.text(S2_freq - 0.55, 0.9, 'M2 S2', color='black')
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # ax1.set_xlabel('Frequency (cpd)')
    ax1.set_ylabel('PSD (K²/cpd)')
    ax1.legend()
    ax1.set_ylim(1e-5, 5)
    ax1.set_xlim(1.5e-1, 100)

    # Right subplot: Zoom in on the peaks (linear scale, x limits 0.5 to 2)
    ax2.plot(F_obs, P_obs, label='Observation', alpha=0.5)
    ax2.plot(F_obs, Peaks, label='Harmonic Analysis')
    # ax2.plot(F_GM, P_GM, label='Garret-Munk', linewidth=2.5)
    ax2.axvline(x=O1_freq, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=K1_freq, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=M2_freq, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=S2_freq, color='black', linestyle='--', alpha=0.5)
    # ax2.axvline(x=f_coriolis_cpd, color='black', linestyle='--', alpha=0.5)
    # ax2.text(f_coriolis_cpd - 0.05, 0.9, 'f', color='black')
    ax2.text(O1_freq-0.06, 0.9, 'O1      K1', color='black')
    ax2.text(S2_freq-0.13, 0.9, 'M2      S2', color='black')
    # Normal (linear) scale on the x-axis with zoom-in limits
    ax2.set_xlabel('Frequency (cpd)')
    ax2.set_ylabel('PSD (K²/cpd)')
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 5)
    ax2.set_xlim(0.8, 2.2)

    plt.tight_layout()
    return fig
