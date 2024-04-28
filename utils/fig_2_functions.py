
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension, _PREFIXES_FACTORS, _LATEX_MU
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class TimeDimension(_Dimension):
    def __init__(self):
        super().__init__("s")
        for prefix, factor in _PREFIXES_FACTORS.items():
            latexrepr = None
            if prefix == "\u00b5" or prefix == "u":
                latexrepr = _LATEX_MU + "s"
            self.add_units(prefix + "s", factor, latexrepr)


def plot_correlations(df_, color, r_position, ax):
    """This function `plot_correlations` is used to create a scatter plot showing the 
    correlation between the 'actual' and 'predicted' values in a DataFrame `df_`. 
    Args:
        df_ (_type_): pandas dataframe containing the actual and predicted values
        color (_type_): scatter plot color
        r_position (_type_): where the correlation value will be plotted
        ax (_type_): object from fig, ax = plt.subplots()
    """
    sns.scatterplot(x='actual', data=df_,
                y='predicted', ax=ax,
                color=color, s=8, alpha=0.4)
    ax.tick_params(axis='both', labelsize=6, width=0.2, length=2.5, pad=1)
    ax.set_ylabel('predicted', fontsize=6, labelpad=1)
    ax.set_xlabel('neural activity', fontsize=6, labelpad=1)
    correlation = df_['actual'].corr(df_['predicted'])
    ax.text(x=r_position[0], y=r_position[1], s=f"r={correlation:.2f}", 
            fontsize=6, transform=ax.transAxes)
    ax.spines[:].set_linewidth(0.3)
    sns.despine()

def plot_three_neurons(frame_start, frame_stop, resp, predictions, reordered_neurons, color, axes, animal='mouse', ylim=None, condition_type=None):
    """used to plot the responses of three neurons over a series of frames

    Args:
        frame_start (_type_): _description_
        frame_stop (_type_): _description_
        resp (_type_): _description_
        predictions (_type_): _description_
        reordered_neurons (_type_): _description_
        color (_type_): _description_
        axes (_type_): _description_
        animal (str, optional): _description_. Defaults to 'mouse'.
        ylim (_type_, optional): _description_. Defaults to None.
        condition_type (_type_, optional): _description_. Defaults to None.
    """
    if animal=='mouse':
        frame_indices = np.arange(frame_start, frame_stop)
    elif animal=='monkey':
        frame_indices = np.arange(16*frame_start, 16*frame_stop)
    n_frames = len(frame_indices)
    n_rows = len(axes)
    for neuron, ax in enumerate(axes.flat):
        ax.plot(frame_indices,resp[frame_indices,reordered_neurons[neuron]], 
                color=color, linewidth=0.6 )
        ax.plot(frame_indices,predictions[frame_indices,reordered_neurons[neuron]], 
                color='red', linewidth=0.6, alpha=0.8 )
        ax.set_ylim(ylim)
        if animal=='monkey':
            if condition_type =='SNR':
                ax.set_xticks(ticks=np.arange(16*frame_start, 16*frame_stop, 16*2), labels=np.arange(frame_start, frame_stop, 2))
                ax.set_ylim(top=ax.get_ylim()[1]+ (0.2*ax.get_ylim()[1]))
            elif condition_type =='SNR_spont':
                ax.set_xticks(np.arange(0,n_frames,24))
                ax.set_xticklabels(np.arange(1,31,3))
        else:
            ax.set_xticks(np.arange(frame_start+5, frame_stop,20))
            ax.set_xlim(frame_start-1, frame_stop+1)
        
        if neuron< n_rows-1:
            # print(neuron)
            ax.set_xticks([])
        else:
            ax.set_xlabel('image presentation no.', fontsize=6, labelpad=1)
        ax.spines[:].set_linewidth(0.3)
        ax.tick_params(axis='both', labelsize=6, width=0.2, length=2.5, pad=1)  
    sns.despine()
    plt.subplots_adjust(hspace=0.2)

def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None
def get_property_dataset_type(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    else:
        return input_string 

def make_mouse_df(mouse_stats_, dataset_types=['ori32','natimg32']):
    """This function `make_mouse_df` is creating a DataFrame from the provided 
    `mouse_stats_` data for different dataset types. It iterates over the dataset types, 
    extracts relevant information for each mouse and area, and then appends this information 
    to a list called `data`. Finally, it creates a DataFrame `df_mouse_all` from the 
    collected data and returns it.

    Args:
        mouse_stats_ (_type_): _description_
        dataset_types (list, optional): _description_. Defaults to ['ori32','natimg32'].

    Returns:
        _type_: _description_
    """
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        else:
            act_type = 'stimulus'
        for mouse, areas_data in mouse_stats_[dataset_type].items():
            mouse_name = extract_mouse_name(mouse)
            for area, values in areas_data.items():
                # Get the split-half correlation values for the current area
                split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
                # Get the SNR values for the current area
                SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
                # Iterate over each pair of split-half correlation, SNR, EV, and null EV values
                for split_half_r, snr, evar, null_evar in zip(split_half_rs, SNRs,values['evars'],values['evars_null']):
                    # Append data for the actual experiment (control_shuffle = False)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':False, 
                    })
                    # Append data for the shuffled experiment (control_shuffle = True)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area,
                        'EV': null_evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                    })
    # Create a DataFrame from the flattened data
    df_mouse_all = pd.DataFrame(data)
    return df_mouse_all

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 

def make_monkey_df(monkey_stats_, dataset_types=['SNR', 'RF_thin', 'RF_large']):
    """
    Create a DataFrame from the provided monkey statistics data for different dataset types.

    Args:
    - monkey_stats_ (_type_): Monkey statistics data.
    - dataset_types (list, optional): List of dataset types. Defaults to ['SNR', 'RF_thin', 'RF_large'].

    Returns:
    - pandas.DataFrame: DataFrame containing the collected monkey data.
    """
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        elif 'RS' in dataset_type:
            act_type = 'lights off'
        else:
            act_type = 'stimulus'
        for date, areas_data in monkey_stats_[dataset_type].items():
            for area, values in areas_data.items():
                # Get the split-half correlation values for the current area
                split_half_rs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['split_half_r']
                # Get the SNR values for the current area
                SNRs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['SNR_meanspont']
                for split_half_r, snr, evar, null_evar in zip(split_half_rs, SNRs,values['evars'],values['evars_null']):
                    # Append data for the actual experiment (control_shuffle = False)
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date':date,
                        'Area': area,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':False, 
                    })
                    # Append data for the shuffled experiment (control_shuffle = True
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date': date,
                        'Area': area,
                        'EV': null_evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                    })
    # Create a DataFrame from the flattened data
    df_monkey_all = pd.DataFrame(data)
    return df_monkey_all