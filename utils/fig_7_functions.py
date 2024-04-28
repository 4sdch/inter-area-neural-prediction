import numpy as np
import sys
from joblib import Parallel, delayed
import time
import pandas as pd

main_dir = ''
func_dir = main_dir + 'utils/'
fig_dir = main_dir + 'results/paper_figures/'
sys.path.insert(0,func_dir)

from macaque_data_functions import get_img_resp_avg_sem
from ridge_regression_functions import get_best_alpha_evars
from macaque_data_functions import get_resps, get_get_condition_type
from stats_functions import get_t_test_stars



all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                      'RS_open':20, 'RS_closed': 20, 
                      'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                      'RS_open':None, 'RS_closed': None, 
                      'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200, 'RF_large_spont':200}

def trial_randomize(istim, seed=None):
    """
    Randomizes trial order while preserving stimulus identity labels.

    Parameters:
    - istim (array-like): Array containing stimulus identity labels for each trial.
    - seed (int, optional): Seed for random number generation. Default is None.

    Returns:
    - shuffled_data (list): Shuffled indices representing the randomized trial order while preserving stimulus identity.
    """
    unique_istims = np.unique(istim)
    num_unique_istims = len(unique_istims)

    # Set the random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    label_values = {}
    for s_idx in range(num_unique_istims):
        s = unique_istims[s_idx]
        loc = np.where(istim == s)[0]
        label_values[s_idx]=list(loc)

    # Shuffle values within each label group
    for label in label_values:
        np.random.shuffle(label_values[label])

    # Reconstruct the shuffled data while maintaining the label order
    shuffled_data = [label_values[label].pop(0) for label in istim]

    return shuffled_data


def get_resps_mini(ref_resp, condition_type, date, ref_on, ref_off, w_size=25, spont_stim_off=300):
    """
    Extracts the neural responses corresponding to a specific time window from the reference response data.

    Args:
        ref_resp (numpy.ndarray): Reference neural response data.
        condition_type (str): Type of experimental condition.
        date (str): Date of the experiment.
        ref_on (int): Start index of the reference time window.
        ref_off (int): End index of the reference time window.
        w_size (int, optional): Size of the time window for averaging. Defaults to 25.
        spont_stim_off (int, optional): Offset for spontaneous activity stimulus. Defaults to 300.

    Returns:
        numpy.ndarray: Neural responses within the specified time window.
    """
    if 'spont' in condition_type:
        chunk_size=int(spont_stim_off/w_size)
    else:
        chunk_size=None
    #reshape the array so that it is shaped (trial_repeats, n_frames_per_repeat, n_neurons)
    chunks = get_img_resp_avg_sem(ref_resp, date, condition_type, get_chunks=True, w_size=w_size, chunk_size=chunk_size)
    chunks_isolated = chunks[:,ref_on:ref_off, :]
    resp_mini = chunks_isolated.reshape(-1,ref_resp.shape[1])
    return resp_mini

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 
    
import random
# Set a seed for reproducibility (optional)
random.seed(17)  # You can use any integer value as the seed
# Generate 10 random seed numbers
seeds = [random.randint(1, 1000) for _ in range(10)]

def get_simil_reli_indices(reli1, reli1_indices, reli2, reli2_indices, seed, verbose=False):
    """This function subsamples the indices of the first group (`reli1_indices`) 
    based on the reliability values of two groups (`reli1` and `reli2`) to ensure 
    similar reliability distributions between the two groups. The subsampling is 
    performed by selecting indices from `reli1_indices` that correspond to reliability 
    values in `reli1` that are close to the reliability values in `reli2`.

    Args:
        reli1 (numpy.ndarray): Array containing reliability values for the first group.
        reli1_indices (numpy.ndarray): Indices corresponding to the first group.
        reli2 (numpy.ndarray): Array containing reliability values for the second group.
        reli2_indices (numpy.ndarray): Indices corresponding to the second group.
        seed (int): Seed value for reproducible random sampling.

    Returns:
        numpy.ndarray: Subsampled indices from the first group.
        numpy.ndarray: Indices from the second group (unchanged).

    Raises:
        None

    """
    new_array1_indices = []

    array1= reli1[reli1_indices]
    array2=reli2[reli2_indices]
    
    for reli_val2 in array2:
        array1_vals = []
        tolerance = 0.001
        count=0
        for a1, reli_val1 in enumerate(array1):
            if np.isclose(reli_val2, reli_val1, atol=tolerance) and a1 not in new_array1_indices:
                count =+1
                array1_vals.append(a1)  
        while count==0:
            if verbose is True:
                print(f'{tolerance} didnt work')
            tolerance *= 2
            for a1, reli_val1 in enumerate(array1):
                if np.isclose(reli_val2, reli_val1, atol=tolerance) and a1 not in new_array1_indices:
                    count =+1
                    array1_vals.append(a1)
        np.random.seed(seed)
        new_array1_indices.append(np.random.choice(array1_vals))
    return reli1_indices[new_array1_indices], reli2_indices

def store_V1_indices(monkey_stats, condition_types = ['SNR','RF_thin','RF_large']):
    """Store V1 indices in monkey statistics.

    This function computes and stores V1 indices in the monkey statistics data. It iterates over each condition type
    and date, retrieves reliability values for both V4 and V1 areas, filters the indices based on reliability
    thresholds for V4, and computes similar reliability indices for V1 using the `get_simil_reli_indices` function.
    The resulting indices are stored in the monkey statistics data under the 'V1_chosen_indices' key.

    Args:
        monkey_stats (dict): Dictionary containing monkey statistics data.
        condition_types (list, optional): List of condition types. Defaults to ['SNR', 'RF_thin', 'RF_large'].

    Returns:
        None
    """
    area='V4'
    area2='V1'
    for condition_type in condition_types:
        for date in monkey_stats[condition_type]:
            reli = monkey_stats[get_property_dataset_type_monkey(condition_type)][date][area]['split_half_r']
            reli2 = monkey_stats[get_property_dataset_type_monkey(condition_type)][date][area2]['split_half_r']
            V4_filtered_indices = np.argwhere(reli > 0.8)[:,0]
            V1_filtered_indices = np.argwhere(reli2 > 0.8)[:,0]
            n_neurons = len(V4_filtered_indices)
            V1_chosen_indices = np.empty((len(seeds), n_neurons), dtype=int)
            for s, seed in enumerate(seeds):
                V1_chosen_indices[s], _= get_simil_reli_indices(reli2, V1_filtered_indices, reli, V4_filtered_indices, seed)
            monkey_stats[condition_type][date][area2]['V1_chosen_indices']=V1_chosen_indices
            
def get_filtered_indices(condition_type, date, ref_area):
    relis = monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4']['split_half_r']
    snrs = monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V4']['SNR_meanspont']
    v4_indices = np.argwhere((relis>0.8)&(snrs >=2))[:,0]
    v4_seed_indices = np.tile(v4_indices, (10,1))
    v1_seed_indices = monkey_stats[get_property_dataset_type_monkey(condition_type)][date]['V1']['V1_chosen_indices']

    if ref_area == 'V4':
        ref_indices = v4_seed_indices
        shift_indices = v1_seed_indices
    else:
        ref_indices = v1_seed_indices
        shift_indices = v4_seed_indices
    return shift_indices, ref_indices



def get_timelag_evars_mini(ref_mini, shift_resp, condition_type, date, ref_on, ref_off, timelag, 
                           ref_area=None, frames_reduced=5, n_splits=10, control_neurons=False, w_size=25,spont_stim_off=300):
    """
    Computes the explained variances for frame subsampled response activity using a time-shifted activity.

    Args:
        ref_mini (numpy.ndarray): Miniaturized reference neural response data.
        shift_resp (numpy.ndarray): Shifted neural response data.
        condition_type (str): Type of experimental condition.
        date (str): Date of the experiment.
        ref_on (int): Start index of the reference time window.
        ref_off (int): End index of the reference time window.
        timelag (int): Time lag for shifting the response.
        ref_area (str, optional): Reference brain area. Defaults to None.
        frames_reduced (int, optional): Number of frames reduced. Defaults to 5.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        control_neurons (bool, optional): Whether to control for neurons. Defaults to False.
        w_size (int, optional): Size of the time window for averaging. Defaults to 25.
        spont_stim_off (int, optional): Offset for spontaneous activity stimulus. Defaults to 300.

    Returns:
        tuple: Tuple containing alpha (ridge regression coefficient) and evars (explained variances).
    """
    shift_mini = get_resps_mini(shift_resp, condition_type, date, int(ref_on+timelag), int(ref_off+timelag), w_size=w_size,spont_stim_off=spont_stim_off)
    # print(shift_mini.shape)
    if control_neurons is True:
        alpha, evars = [], []
        shift_indices, ref_indices = get_filtered_indices(condition_type, date, ref_area)
        results = Parallel(n_jobs=-1)(delayed(get_best_alpha_evars)(shift_mini[:, shift_indices[s]], 
                                                                    ref_mini[:,ref_indices[s]], n_splits=n_splits, frames_reduced=frames_reduced) for s in range(10))
        for al, ev in results:
            alpha.append(al)
            evars.append(ev)
        alpha = np.array(alpha)
        evars = np.array(evars)
    else:
        alpha, evars = get_best_alpha_evars(shift_mini, ref_mini, n_splits=n_splits, frames_reduced=frames_reduced)
    return alpha, evars

def get_timelag_evars_ref_time(ref_resp, shift_resp, condition_type, date, ref_on, 
                            ref_off, ref_dur, ref_area=None, frames_reduced=5, n_splits=10, 
                            control_neurons=False, w_size=25,spont_stim_off=300):
    """
    Computes the explained variances for different time lags between reference and shifted responses.

    Args:
        ref_resp (numpy.ndarray): Reference neural response data.
        shift_resp (numpy.ndarray): Shifted neural response data.
        condition_type (str): Type of experimental condition.
        date (str): Date of the experiment.
        ref_on (int): Start index of the reference time window.
        ref_off (int): End index of the reference time window.
        ref_dur (int): Duration of the reference time window.
        ref_area (str, optional): Reference brain area. Defaults to None.
        frames_reduced (int, optional): Number of frames reduced. Defaults to 5.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        control_neurons (bool, optional): Whether to control for neurons. Defaults to False.
        w_size (int, optional): Size of the time window for averaging. Defaults to 25.
        spont_stim_off (int, optional): Offset for spontaneous activity stimulus. Defaults to 300.

    Returns:
        tuple: Tuple containing alpha (ridge regression coefficient) and evars (explained variances) for each time lag.
    """
    if 'spont' in condition_type:
        real_dur=int(300/w_size)
    elif 'RF' in condition_type:
        real_dur = int(1000/w_size)
    elif 'SNR' in condition_type:
        real_dur = int(400/w_size)

    timelags = np.arange(-ref_on, -ref_on+(real_dur - ref_dur)+1)
    # print(timelags)
    ref_mini = get_resps_mini(ref_resp, condition_type, date, ref_on, ref_off, w_size=w_size,spont_stim_off=spont_stim_off)
    all_alphas, all_evars = [],[]
    results = Parallel(n_jobs=-1)(delayed(get_timelag_evars_mini)(ref_mini, shift_resp, condition_type, date,
                                                                ref_on, ref_off, timelag, ref_area,
                                                                frames_reduced,n_splits, control_neurons,w_size=w_size,spont_stim_off=spont_stim_off) for timelag in timelags)
    for alpha, evar in results:
        all_alphas.append(alpha)
        all_evars.append(evar)
    alphas_array = np.array(all_alphas)
    evars_array = np.array(all_evars)
    
    return alphas_array, evars_array


all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_eyes_open':20, 'RS_eyes_closed': 20, 
                    'RF_thin':25, 'RF_large':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_eyes_open':None, 'RS_eyes_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200,
                    'RF_large_spont':200}

def get_ref_shift(resp_V1, resp_V4, ref_area):
    if ref_area =='V1':
        ref_resp = resp_V1
        shift_resp = resp_V4
    elif ref_area=='V4':
        ref_resp = resp_V4
        shift_resp = resp_V1
    return ref_resp, shift_resp

def get_refons_refoffs(ref_duration, w_size, condition_type, spont_stim_off=300):
    ref_dur = int(ref_duration/w_size)
    if 'spont' in condition_type:
        real_dur = int(spont_stim_off/w_size)
    elif 'RF' in condition_type:
        real_dur = int(1000/w_size)
    elif 'SNR' in condition_type:
        real_dur = int(400/w_size)
    ref_ons = np.arange((real_dur - ref_dur)+1, dtype=int)
    ref_offs = ref_ons + int(ref_dur)
    return ref_ons, ref_offs

def process_timelag_shenanigans(condition_type, date, ref_area, 
                                ref_duration, monkey_stats_timelags,  
                                w_size=25, control_neurons=False,
                                n_splits=10, spont_stim_off=300):
    """
    Processes the time lag predictions by computing explained variances for different time lags.

    Args:
        condition_type (str): Type of experimental condition.
        date (str): Date of the experiment.
        ref_area (str): Reference brain area.
        ref_duration (int): Duration of the reference time window.
        monkey_stats_timelags (dict): Dictionary to store time lag statistics.
        w_size (int, optional): Size of the time window for averaging. Defaults to 25.
        control_neurons (bool, optional): Whether to control for neurons. Defaults to False.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        spont_stim_off (int, optional): Offset for spontaneous activity stimulus. Defaults to 300.
    """
    start_time = time.time()
    ref_dur = int(ref_duration/w_size)
    if 'RF' in condition_type:
        real_dur = int(1000/w_size)
    elif 'SNR_spont' in condition_type:
        real_dur= int(spont_stim_off/w_size)
        
    elif 'SNR' in condition_type:
        real_dur = int(400/w_size)
    
    frames_reduced=int(np.round((all_frames_reduced[condition_type])*25/w_size))
    if ref_dur < frames_reduced:
        frames_reduced = int(ref_dur -1)

    initial_stim_off = all_ini_stim_offs[condition_type]

    ref_ons = np.arange((real_dur - ref_dur)+1, dtype=int)
    ref_offs = ref_ons + int(ref_dur)


    resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(condition_type), 
                                    date=date, w_size=w_size, stim_on=0, stim_off=initial_stim_off, spont_stim_off=spont_stim_off)
    # print(resp_V4.shape)
    ref_resp, shift_resp = get_ref_shift(resp_V1, resp_V4, ref_area)
    
    results = Parallel(n_jobs=-1)(delayed(get_timelag_evars_ref_time)(ref_resp=ref_resp,shift_resp=shift_resp,condition_type=condition_type, 
                                                                    date=date,ref_on=ref_on,ref_off=ref_off, ref_dur=ref_dur, 
                                                                    ref_area=ref_area, control_neurons=control_neurons,
                                                            frames_reduced=frames_reduced, n_splits=n_splits,w_size=w_size) for ref_on, ref_off in zip(ref_ons, ref_offs))                                                             
    if control_neurons is True:
        for t, (alpha, evar) in enumerate(results):
            monkey_stats_timelags[condition_type][date][ref_area][f'timelag_evars_{ref_ons[t]}_{ref_offs[t]}_all_seeds']=evar
            monkey_stats_timelags[condition_type][date][ref_area][f'timelag_alphas_{ref_ons[t]}_{ref_offs[t]}_all_seeds']=alpha
    else:
        for t, (alpha, evar) in enumerate(results):
            monkey_stats_timelags[condition_type][date][ref_area][f'timelag_evars_{ref_ons[t]}_{ref_offs[t]}_all_neurons']=evar
            monkey_stats_timelags[condition_type][date][ref_area][f'timelag_alphas_{ref_ons[t]}_{ref_offs[t]}_all_neurons']=alpha
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = (end_time - start_time)/60
    print(f'yay! date {date} for {condition_type} is completed')
    print(f'Took {elapsed_time:.4f} minutes to complete')
    
## plotting functions


def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None
def get_property_dataset_type(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    else:
        return input_string 
def make_mouse_df_time(mouse_stats_, dataset_types=['ori32','natimg32']):
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        else:
            act_type = 'stimulus'
        for mouse, areas_data in mouse_stats_[dataset_type].items():
            mouse_name = extract_mouse_name(mouse)
            for area, values in areas_data.items():
                if area=='L23':
                    direction = 'L4→L2/3'
                    area_ = 'L2/3'
                else:
                    direction = 'L2/3→L4'
                    area_=area
                split_half_rs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['split_half_r']
                SNRs = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['SNR_meanspont']
                one_vs_rests = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['1_vs_rest_r']
                trial_shuffle_evars = mouse_stats_[get_property_dataset_type(dataset_type)][mouse][area]['evar_shuffled_istims']
                for n, (split_half_r, snr,max_corr_val, onevsrest, evar, null_evar, trial_shuff_ev) in enumerate(zip(split_half_rs, SNRs,values['max_corr_val'], one_vs_rests, values['evars'],values['evars_null'], trial_shuffle_evars)):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'split-half r': split_half_r,
                        'max r² val':np.square(max_corr_val),
                        '1-vs-rest r²': np.square(onevsrest),
                        'control_shuffle':False, 
                        'Neuron':n,
                        'EV shuffled':trial_shuff_ev
                    })
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'Direction':direction,
                        'EV': null_evar,
                        'SNR': snr,
                        'split-half r': split_half_r,
                        'max r² val':np.square(max_corr_val),
                        '1-vs-rest r²': np.square(onevsrest),
                        'control_shuffle':True,
                        'Neuron':n,
                        'EV shuffled':trial_shuff_ev
                    })
    # Create a DataFrame from the flattened data
    df_mouse_all = pd.DataFrame(data)
    return df_mouse_all

def plot_null_line(df_, neuron_property, ax, color='blue', label='shuffle\ncontrol IQR'):
    data = df_[neuron_property]
    per_25 = np.percentile(data.dropna().values, 25)
    per_75 = np.percentile(data.dropna().values, 75)
    ax.axhspan(per_25, per_75, alpha=0.1, color=color, label=label,
            linewidth=0)

def get_property_dataset_type_monkey(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    elif 'RS' in input_string:
        return 'SNR'
    else:
        return input_string 

def make_monkey_df_time(monkey_stats_, dataset_types=['SNR', 'RF_thin', 'RF_large']):
    data = []
    for dataset_type in dataset_types:
        if 'spont' in dataset_type:
            act_type = 'gray screen'
        elif dataset_type=='RS':
            act_type = 'lights off'
        elif dataset_type =='RS_open':
            act_type = 'lights off\neyes open'
        elif dataset_type =='RS_closed':
            act_type = 'lights off\neyes closed'
        else:
            act_type = 'stimulus'
        for date, areas_data in monkey_stats_[dataset_type].items():
            for area, values in areas_data.items():
                if area=='V4':
                    direction = 'V1→V4'
                else:
                    direction = 'V4→V1'
                split_half_rs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['split_half_r']
                SNRs = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['SNR_meanspont']
                one_vs_rests = monkey_stats_[get_property_dataset_type_monkey(dataset_type)][date][area]['1_vs_rest_r']
                evars = values['evars']
                evars_null = values['evars_null']
                
                for split_half_r, snr,max_corr_val,onevsrest, evar, null_evar, shuffle_trial_ev in zip(split_half_rs, SNRs,values['max_corr_val'],one_vs_rests,evars,evars_null, values['evar_shuffled_istims']):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date':date,
                        'Area': area,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'split-half r': split_half_r,
                        'max r² val':np.square(max_corr_val),
                        '1-vs-rest r²': np.square(onevsrest),
                        'control_shuffle':False, 
                        'EV shuffled':shuffle_trial_ev,
                        'control shuffle EV':null_evar,
                    })
    # Create a DataFrame from the flattened data
    df_monkey_all = pd.DataFrame(data)
    return df_monkey_all


def extract_condition(input_string):
    if 'spont' in input_string:
        spont_ = '_spont'
        return input_string.replace('_spont','')
    else:
        return input_string
def get_refons_refoffs(ref_duration, w_size, condition_type, spont_stim_off=300):
    ref_dur = int(ref_duration/w_size)
    if 'spont' in condition_type:
        real_dur = int(spont_stim_off/w_size)
    elif 'RF' in condition_type:
        real_dur = int(1000/w_size)
    elif 'SNR' in condition_type:
        real_dur = int(400/w_size)
    ref_ons = np.arange((real_dur - ref_dur)+1, dtype=int)
    ref_offs = ref_ons + int(ref_dur)
    return ref_ons, ref_offs

## plot it somehow lolol
def make_df_timelags(monkey_stats_timelags, condition_type,ref_area, 
                    ref_ons,ref_offs, ref_duration, control_neurons=False, w_size=25,spont_stim_off=300):
    if 'spont' in condition_type:
        real_dur = int(spont_stim_off/w_size)
        act_type = 'gray screen'
    elif 'SNR' in condition_type:
        real_dur = int(400/w_size)
        act_type = 'stimulus'
    ref_dur = int(ref_duration/w_size)
    data=[]

    for date, areas_data in monkey_stats_timelags[condition_type].items():
        relis = monkey_stats_timelags[get_property_dataset_type_monkey(condition_type)][date][ref_area]['split_half_r']
        snrs = monkey_stats_timelags[get_property_dataset_type_monkey(condition_type)][date][ref_area]['SNR_meanspont']
        if ref_area=='V4':
            pred_label='V1→V4'
            v4_indices = np.argwhere((relis>0.8)&(snrs >=2))[:,0]
            seed_indices = np.concatenate(np.tile(v4_indices, (10,1)))
            # print(len(v4_indices), len(np.unique(v4_indices)))
        elif ref_area=='V1':
            pred_label='V4→V1'
            seed_indices = np.concatenate(monkey_stats_timelags[extract_condition(condition_type)][date]['V1']['V1_chosen_indices'])
        for ref_on, ref_off in zip(ref_ons, ref_offs):
            values = areas_data[ref_area]
            if control_neurons is True:
                timelag_evars = values[f'timelag_evars_{ref_on}_{ref_off}_all_seeds']
            else:
                timelag_evars = values[f'timelag_evars_{ref_on}_{ref_off}_all_neurons']
            timelags = np.arange(-ref_on, -ref_on+(real_dur - ref_dur)+1)
            for t, timelag in enumerate(timelags): 
                if control_neurons is True:
                    evars = np.concatenate(timelag_evars[t])
                    lag0evars = np.concatenate(timelag_evars[np.argwhere(timelags==0)[0,0]])
                    permutations = np.concatenate([np.ones([int(len(seed_indices)/10)],dtype=int)*count for count in range(10)])
                    # print(permutations.shape)
                    for n, evar in enumerate(evars):
                        data.append({
                            'Date': date,
                            'Area': ref_area,
                            'EV': evar,
                            'Offset(ms)': timelag*25,
                            'Direction':pred_label,
                            'Ref_Times': f'{int(ref_on*25)}:{int(ref_off*25)}',
                            'Mean_Norm_EV':evar/np.nanmean(lag0evars),
                            'SNR': snrs[seed_indices[n]],
                            'split-half r': relis[seed_indices[n]],
                            'Permutation': permutations[n],
                            'Neuron':seed_indices[n],
                        })
                else:
                    lag0evars = timelag_evars[np.argwhere(timelags==0)[0,0]]
                    for n, (evar, reli, snr) in enumerate(zip(timelag_evars[t], 
                                        relis, 
                                        snrs)):
                        if ref_area=='V4':
                            pred_label='V1→V4'
                        elif ref_area=='V1':
                            pred_label='V4→V1'
                        data.append({
                            'Date': date,
                            'Area': ref_area,
                            'EV': evar,
                            'Offset(ms)': timelag*25,
                            'SNR': snr,
                            'split-half r': reli,
                            'Direction':pred_label,
                            'Ref_Times': f'{int(ref_on*25)}:{int(ref_off*25)}',
                            'Mean_Norm_EV':evar/np.nanmean(lag0evars),
                            'Neuron':seed_indices[n],
                        })
    # Create a DataFrame from the flattened data
    df = pd.DataFrame(data)
    return df

### supplemental plotting

def make_raw_data_df(condition_type,date,timebin, time_chunk=10000):
    get_condition_type = get_get_condition_type(condition_type)
    area='V4'
    area2="V1"
    resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, 
                                    w_size=timebin, stim_on=0, stim_off=all_ini_stim_offs[condition_type],raw_resp=True)
    
    resp_V4 = resp_V4[:int(np.ceil(time_chunk/timebin))]
    resp_V1 = resp_V1[:int(np.ceil(time_chunk/timebin))]
    
    data = []
    for t in range(resp_V4.shape[0]):
        for val in resp_V4[t]:
            data.append({
                        'Dataset_Type': condition_type,
                        'Date': date,
                        'Area': area,
                        'timebin':timebin,
                        'MUAe':val,
                        'time':t*timebin,
                    })
    for t in range(resp_V1.shape[0]):
        for val in resp_V1[t]:
            data.append({
                        'Dataset_Type': condition_type,
                        'Date': date,
                        'Area': area2,
                        'timebin':timebin,
                        'MUAe':val,
                        'time':t*timebin,
                    })
    return data

def add_stars_2_sets(df_, neuron_property, x, x_order, hue, ax, fontsize=7, height1=0.97, height2=0.97, perm_t=True, perm_type='ind', hierarchical=False, mouse_or_date='Mouse Name', central_tendency='median'):
    stars1 = get_t_test_stars(df_[df_[x]==x_order[0]], hue, neuron_property, perm_t=perm_t, perm_type=perm_type, hierarchical=hierarchical, mouse_or_date=mouse_or_date, central_tendency='median')
    stars2 = get_t_test_stars(df_[df_[x]==x_order[1]], hue, neuron_property, perm_t=perm_t, perm_type=perm_type,hierarchical=hierarchical, mouse_or_date=mouse_or_date, central_tendency='median')
    if stars1 == 'n.s.':
        height1 = height1 + 0.02
        fontsize1 = fontsize*0.9
        color1='#C0C0C0'
    else:
        fontsize1 = fontsize
        color1='black'
        
    if stars2 == 'n.s.':
        height2 = height2 + 0.02
        fontsize2 = fontsize*0.9
        color2='#C0C0C0'
    else:
        fontsize2 = fontsize
        color2='black'
    
    ax.text(0.25, height1, stars1, ha='center', va='center', fontsize=fontsize1, transform=ax.transAxes, color=color1)
    ax.text(0.75, height2, stars2, ha='center', va='center', fontsize=fontsize2, transform=ax.transAxes, color=color2)