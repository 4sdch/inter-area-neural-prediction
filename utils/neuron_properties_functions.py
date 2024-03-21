# Description: This file contains functions for analyzing neural data, specifically for mouse and monkey datasets

main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/inter_areal_predictability/'
func_dir = main_dir + 'utils/'

import sys
sys.path.insert(0,func_dir)

main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/'


import mouse_data_functions as cs
import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from macaque_data_functions import get_img_resp_avg_sem, get_resps, get_get_condition_type
import scipy
from ridge_regression_functions import get_best_alpha_evars
import time

def create_empty_mouse_stats_dict(main_dir):
    """
    Create an empty dictionary to store mouse statistics.

    Args:
    - main_dir (str): Path to the main directory containing data.

    Returns:
    - mouse_stats (dict): Empty dictionary to store mouse statistics.
    """
    mouse_stats={}
    for dataset_type in ['natimg32','ori32']:
        mouse_stats[dataset_type]={}
        mouse_stats[f'{dataset_type}_spont']={}
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = mt.filenames
        for mouse in mousenames:
            mouse_stats[dataset_type][mouse]={}
            mouse_stats[dataset_type][mouse]['L23']={}
            mouse_stats[dataset_type][mouse]['L4']={}
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('spont', mouse)
            if len(resp_L1)<1000:
                continue
            mouse_stats[f'{dataset_type}_spont'][mouse]={}
            mouse_stats[f'{dataset_type}_spont'][mouse]['L23']={}
            mouse_stats[f'{dataset_type}_spont'][mouse]['L4']={}
    return mouse_stats

def get_SNR_all_mice(main_dir, mouse_stats):
    """
    Compute Signal to Noise Ratio (SNR) for all mice.
    signal to noise ratio is calculated using the average 
    activity in response to stimuli over the average activity
    in response to a gray screen presentation.
    
    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.

    Returns:
    - None
    """
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            mt.mt = mt.mts[mouse]
            resp, spont = mt.add_preprocessing() #retrieve raw activity of all neurons. 
            L1indices, L23indices, L2indices, L3indices, L4indices=mt.get_L_indices() # gets the neuron indices that belong to specific layers
            SNR_mean_over_spont= np.mean(resp, axis=0)/np.mean(spont, axis=0) # do simple resp over spont
            mouse_stats[dataset_type][mouse]['L23']['SNR_meanspont'] = SNR_mean_over_spont[L23indices]
            mouse_stats[dataset_type][mouse]['L4']['SNR_meanspont'] = SNR_mean_over_spont[L4indices]
                        
            
def get_split_half_mean_mouse_seed(s_idx, unique_istims, resp, istim,seed=None):
    """
    Compute split-half reliability for a given seed and stimulus index.

    Args:
    - s_idx (int): Index of the stimulus.
    - unique_istims (numpy.ndarray): Array containing unique stimulus indices.
    - resp (numpy.ndarray): Array containing neural responses.
    - istim (numpy.ndarray): Array containing stimulus indices.
    - seed (int): Seed for random number generation.

    Returns:
    - means_half1 (numpy.ndarray): Mean of responses for the first half.
    - means_half2 (numpy.ndarray): Mean of responses for the second half.
    """
    s = unique_istims[s_idx]
    loc = np.where(istim == s)[0]
    if len(loc) > 1:
        # Randomly split the loc array into two halves
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(loc)
        half_size = len(loc) // 2
        loc_half1 = loc[:half_size]
        loc_half2 = loc[half_size:half_size*2]

        # Compute means for both halves and all neurons at once
        means_half1 = np.nanmean(resp[loc_half1], axis=0)
        means_half2 = np.nanmean(resp[loc_half2], axis=0)
    return means_half1, means_half2

def get_split_half_r_mouse(istim, resp, seed=None):
    """
    Compute split-half reliability for all neurons.

    Args:
    - istim (numpy.ndarray): Array containing stimulus indices.
    - resp (numpy.ndarray): Array containing neural responses.
    - seed (int): Seed for random number generation.

    Returns:
    - scsb (numpy.ndarray): Split-half reliability values for each neuron.
    """
    unique_istims = np.unique(istim)
    num_unique_istims = len(unique_istims)
    num_neurons = resp.shape[1]

    scsb = np.zeros(num_neurons)  # Initialize the results array
    x = np.empty((0, num_neurons))  # Initialize x as an empty 2D array
    y = np.empty((0, num_neurons))  # Initialize y as an empty 2D array

    x, y=[],[]
    results = Parallel(n_jobs=-1)(delayed(get_split_half_mean_mouse_seed)(s_idx, unique_istims,resp, istim, seed) for s_idx in range(num_unique_istims))

    for x_, y_ in results:
        x.append(x_)
        y.append(y_)
    
    x = np.array(x)
    y=np.array(y)

    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    scsb = correlations*2/(1+correlations)

    return scsb

def get_max_corr_vals_all_mice(main_dir, mouse_stats,remove_pcs=False):
    """
    Compute maximum correlation values for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        rem_pc = ''
        if remove_pcs is True:
            rem_pc = '_removed_32_pcs'
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse, removed_pc=remove_pcs)
            connx_matrix = np.corrcoef(resp_L23.T, resp_L4.T)
            l23_l4_connx = connx_matrix[:resp_L23.shape[1], resp_L23.shape[1]:]
            mouse_stats[dataset_type][mouse]['L23']['max_corr_val' + rem_pc]=np.nanmax(np.abs(l23_l4_connx), axis=1)
            mouse_stats[dataset_type][mouse]['L4']['max_corr_val' + rem_pc]=np.nanmax(np.abs(l23_l4_connx), axis=0)
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('spont', mouse)
            if resp_L1.shape[0]<1000:
                continue
            connx_matrix = np.corrcoef(resp_L23.T, resp_L4.T)
            l23_l4_connx = connx_matrix[:resp_L23.shape[1], resp_L23.shape[1]:]
            mouse_stats[dataset_type + '_spont'][mouse]['L23']['max_corr_val']=np.nanmax(np.abs(l23_l4_connx), axis=1)
            mouse_stats[dataset_type + '_spont'][mouse]['L4']['max_corr_val']=np.nanmax(np.abs(l23_l4_connx), axis=0)

def get_split_half_r_all_mice(main_dir, mouse_stats, remove_pcs=False):
    """
    Compute split-half reliability for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    rem_pc=''
    if remove_pcs is True:
        rem_pc='_removed_32_pcs'
    for dataset_type in ['natimg32','ori32']:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type)
        mousenames = list(mouse_stats[dataset_type].keys())
        for mouse in mousenames:
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity('resp', mouse, removed_pc=remove_pcs)
            istim = mt.istim
            mouse_stats[dataset_type][mouse]['L23']['split_half_r'+ rem_pc]=get_split_half_r_mouse(istim, resp_L23)
            mouse_stats[dataset_type][mouse]['L4']['split_half_r'+ rem_pc]=get_split_half_r_mouse(istim, resp_L4)

def get_evars_all_mice(main_dir, mouse_stats, activity_type='resp',n_splits=10, frames_to_reduce=5,
                        control_shuffle=False, remove_pcs=False):
    """
    Compute explained variance for all mice.

    Args:
    - main_dir (str): Path to the main directory containing data.
    - mouse_stats (dict): Dictionary containing mouse statistics.
    - activity_type (str): Type of neural activity ('resp' or 'spont').
    - n_splits (int): Number of splits for cross-validation.
    - frames_to_reduce (int): Number of frames to reduce.
    - control_shuffle (bool): Flag indicating whether to shuffle for control.
    - remove_pcs (bool): Flag indicating whether to remove principal components.

    Returns:
    - None
    """
    start_time = time.time()
    alpha_unique_options = [5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7]
    area = 'L23'
    area2='L4'
    dataset_types=['ori32','natimg32']
    control_con = ''
    rem_pc = ''
    spont_con = ''
    if control_shuffle is True:
        control_con = '_null'
    if remove_pc is True:
        rem_pc = '_removed_32_pcs'
    if activity_type =='spont':
        spont_con = '_spont'
    for dataset_type in dataset_types:
        mt = cs.mt_retriever(main_dir, dataset_type=dataset_type) #retrieves neural activity stored in data
        mouse_names = mt.filenames
        for mouse in mouse_names:
            
            resp_L1, resp_L23, resp_L2, resp_L3, resp_L4 = mt.retrieve_layer_activity(activity_type, mouse,removed_pc=remove_pcs)
            if resp_L1.shape[0]<1000:
                # there are some gray screen activity datasets that are too small to fit
                continue      
            alpha, evars = get_best_alpha_evars(resp_L4, resp_L23, n_splits=n_splits, 
                                                frames_reduced=frames_to_reduce, 
                                                alphas=alpha_unique_options)
            alpha2, evars2 = get_best_alpha_evars(resp_L23, resp_L4, n_splits=n_splits, 
                                                frames_reduced=frames_to_reduce, 
                                                alphas=alpha_unique_options)

            mouse_stats[dataset_type + spont_con][mouse][area]['evars' + control_con + rem_pc]=evars
            mouse_stats[dataset_type + spont_con][mouse][area2]['evars' + control_con + rem_pc]=evars2
            mouse_stats[dataset_type + spont_con][mouse][area]['alpha' + control_con + rem_pc]=alpha
            mouse_stats[dataset_type + spont_con][mouse][area2]['alpha' + control_con + rem_pc]=alpha2
        print(dataset_type, 'done')
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'yay! it took {elapsed_time:.2f} minutes to finish all dataset types!')


condition_types =['SNR', 'SNR_spont', 'RS', 'RS_open', 'RS_closed', 
                  'RF_thin', 'RF_large', 'RF_thin_spont', 'RF_large_spont']
def get_dates(condition_type):
    if 'SNR' in condition_type or 'RS' in condition_type:
        return ['090817', '100817', '250717']
    elif 'large' in condition_type:
        return ['260617']
    else:
        return ['280617']

def create_empty_monkey_stats_dict():
    """
    Create an empty dictionary for monkey statistics.

    Returns:
    - monkey_stats (dict): Empty dictionary for monkey statistics.
    """
    monkey_stats={}
    for dataset_type in ['SNR', 'SNR_spont','RS','RS_open','RS_closed','RF_thin','RF_thin_spont','RF_large','RF_large_spont']:
        monkey_stats[dataset_type]={}
        dates = get_dates(dataset_type)
        for date in dates:
            monkey_stats[dataset_type][date]={}
            monkey_stats[dataset_type][date]['V4']={}
            monkey_stats[dataset_type][date]['V1']={}
    return monkey_stats


def get_split_half_shape_monkey_seed(resp_array, date, condition_type, subsample_size=20):
    
    binned_epochs = get_img_resp_avg_sem(resp_array, date, condition_type=condition_type, get_chunks=True)
    all_epoch_indices = np.arange(len(binned_epochs))
    epoch_indices = np.random.choice(all_epoch_indices, subsample_size)
    half_size = len(epoch_indices) // 2
    x=binned_epochs[epoch_indices[:half_size]].mean(axis=0)
    y=binned_epochs[epoch_indices[half_size:half_size*2]].mean(axis=0)
    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    return correlations*2/(1+correlations)

def get_split_half_r_monkey(resp_array, date, condition_type, n_perms=100):
    
    results = Parallel(n_jobs=-1)(delayed(get_split_half_shape_monkey_seed)(resp_array, date, condition_type) for p in range(n_perms))
    v_elec_mean_rs = np.array(results).mean(axis=0)
    return v_elec_mean_rs

def get_split_half_shape_monkey_RF_seed(resp_array, cond_labels, date, condition_type):
    all_x = []
    all_y = []

    binned_epochs = get_img_resp_avg_sem(resp_array, date, condition_type=condition_type, get_chunks=True)
    binned_labels = cond_labels[:,0,0]

    for cond_num in range(len(np.unique(binned_labels))):
        stim_epochs = binned_epochs[np.argwhere(binned_labels==cond_num)[:, 0]]
        epoch_indices = np.arange(len(stim_epochs))
        np.random.shuffle(epoch_indices)
        half_size = len(epoch_indices) // 2
        all_x.append(stim_epochs[epoch_indices[:half_size]].mean(axis=0))
        all_y.append(stim_epochs[epoch_indices[half_size:half_size*2]].mean(axis=0))
    
    x = np.concatenate(all_x, axis=0)
    y = np.concatenate(all_y, axis=0)

    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    return correlations*2/(1+correlations)

def get_split_half_r_monkey_RF(resp_array, cond_labels, date, condition_type, n_perms=100):
    v_elec_rs = []
    results = Parallel(n_jobs=-1)(delayed(get_split_half_shape_monkey_RF_seed)(resp_array, cond_labels, date, condition_type) for p in range(n_perms))
    for corr in results:
        v_elec_rs.append(corr)
    return np.array(v_elec_rs).mean(axis=0)

#depending on the dataset type, there are different times of autocorrelation to mitigate
all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                      'RS_open':20, 'RS_closed': 20, 
                      'RF_thin':25, 'RF_large':25, 'RF_thin_spont':25, 'RF_large_spont':25}
#different stimulus presentaion types have different durations
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 300, 'RS': None,
                      'RS_open':None, 'RS_closed': None, 
                      'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':300, 
                      'RF_large_spont':300}

def get_split_half_r_monkey_all_dates(monkey_stats, w_size=25):
    """
    Compute split-half reliability for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    for dataset_type in ['RF_large','SNR','RF_thin']:
        dates = get_dates(dataset_type)
        for date in dates:
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], get_RF_labels=True)
                monkey_stats[dataset_type][date][area]['split_half_r']=get_split_half_r_monkey_RF(resp_V4, cond_labels, date,dataset_type)
                monkey_stats[dataset_type][date][area2]['split_half_r']=get_split_half_r_monkey_RF(resp_V1, cond_labels, date, dataset_type)
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type])
                monkey_stats[dataset_type][date][area]['split_half_r']=get_split_half_r_monkey(resp_V4, date, dataset_type)
                monkey_stats[dataset_type][date][area2]['split_half_r']=get_split_half_r_monkey(resp_V1, date, dataset_type)

def get_SNR_monkey(binned_resp, binned_spont):
    baseline_stack_avg = np.mean(binned_spont, axis=0)
    baseline_avg = np.mean(baseline_stack_avg, axis=0)
    baseline_std = np.std(baseline_stack_avg, axis=0)
    
    MUA_avg = np.mean(binned_resp, axis=0)
    window = 20  # hard-coded
    mask = np.ones((window)) / window
    MUA_sm = scipy.ndimage.convolve1d(MUA_avg, mask, axis=0)
    MUA_max = np.max(MUA_sm, axis=0)
    
    # Calculate channel Signal to Noise Ratio (SNR)
    SNR = (MUA_max - baseline_avg) / baseline_std
    return SNR

def get_SNR_monkey_all_dates(monkey_stats, w_size=1):
    """
    Compute SNR for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    for dataset_type in ['SNR','RF_large','RF_thin']:
        dates = get_dates(dataset_type)
        for date in dates:
            resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], raw_resp=True)
            spont_V4, spont_V1 =get_resps(condition_type=get_get_condition_type(dataset_type)+'_spont', date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], raw_resp=True)
            monkey_stats[dataset_type][date][area]['SNR_meanspont']=get_SNR_monkey(resp_V4, spont_V4)
            monkey_stats[dataset_type][date][area2]['SNR_meanspont']=get_SNR_monkey(resp_V1, spont_V1)

def get_max_corr_vals_monkey_all_dates(monkey_stats, w_size=25):
    area='V4'
    area2='V1'
    for dataset_type in monkey_stats:
        for date in monkey_stats[dataset_type]:
            resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type])
            connx_matrix = np.corrcoef(resp_V4.T, resp_V1.T)
            v4_v1_connx = connx_matrix[:resp_V4.shape[1], resp_V4.shape[1]:]
            monkey_stats[dataset_type][date][area]['max_corr_val'] = np.nanmax(np.abs(v4_v1_connx), axis=1)
            monkey_stats[dataset_type][date][area2]['max_corr_val'] = np.nanmax(np.abs(v4_v1_connx), axis=0)

### monkey

def get_1_vs_all_scsb_monkey_1trial(trial_no, binned_epochs):
    # Compute means for both halves and all neurons at once
    x= binned_epochs[trial_no]
    bulk_half = np.delete(binned_epochs, trial_no, axis=0)
    y = np.nanmean(bulk_half, axis=0)
    
    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    # no corrections

    return correlations

from joblib import Parallel, delayed

def get_1_vs_rest_r_monkey(binned_epochs):
    n_trials = len(binned_epochs)
    results = Parallel(n_jobs=-1)(delayed(get_1_vs_all_scsb_monkey_1trial)(trial_no, binned_epochs) for trial_no in range(n_trials))
    scsbs = []
    for sc in results:
        scsbs.append(sc)
    scsb = np.mean(np.array(scsbs), axis=0)

    return scsb

def get_1_vs_all_scsb_monkey_RF_1trial(binned_labels, binned_epochs, trial_no, trial_avg=False):
    x, y = [],[]
    for cond_num in range(len(np.unique(binned_labels))):
        loc = np.argwhere(binned_labels==cond_num)[:, 0]
        x.append(binned_epochs[loc[trial_no]])
        y.append(np.nanmean(binned_epochs[np.delete(loc, trial_no)],axis=0))

    if trial_avg is True:
        x = np.array(x).mean(axis=1)
        y = np.array(y).mean(axis=1)
    else:
        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

    correlations = np.array([stats.pearsonr(x[:,neuron], y[:,neuron])[0] for neuron in range(x.shape[1])])
    # no correction 
    return correlations

def get_min_trials(binned_labels):
    trial_nos=[]
    for cond_num in range(len(np.unique(binned_labels))):
        loc = np.argwhere(binned_labels==cond_num)[:, 0]
        trial_nos.append(len(loc))
    return min(trial_nos)

from macaque_data_functions import get_img_resp_avg_sem
def get_1_vs_rest_r_monkey_RF(resp_array, cond_labels, date, condition_type, trial_avg=False):
    scsbs = []

    binned_epochs = get_img_resp_avg_sem(resp_array, date, condition_type=condition_type, get_chunks=True)
    binned_labels = cond_labels[:,0,0]

    n_trials = get_min_trials(binned_labels)
    results = Parallel(n_jobs=-1)(delayed(get_1_vs_all_scsb_monkey_RF_1trial)(binned_labels, binned_epochs, trial_no, trial_avg) for trial_no in range(n_trials))

    for sc in results:
        scsbs.append(sc)
    
    return np.mean(np.array(scsbs), axis=0)

from macaque_data_functions import get_get_condition_type, get_resps, get_img_resp_avg_sem
def get_one_vs_rest_r_monkey_all_dates(monkey_stats, w_size=25):
    """
    Compute 1 vs. rest reliability for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    area='V4'
    area2='V1'
    for dataset_type in ['RF_large','SNR','RF_thin']:
        dates = get_dates(dataset_type)
        for date in dates:
            if 'RF' in dataset_type:
                resp_V4, resp_V1, cond_labels =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type], get_RF_labels=True)
                monkey_stats[dataset_type][date][area]['1_vs_rest_r']=get_1_vs_rest_r_monkey_RF(resp_V4, cond_labels, date, dataset_type)
                monkey_stats[dataset_type][date][area2]['1_vs_rest_r']=get_1_vs_rest_r_monkey_RF(resp_V1, cond_labels, date, dataset_type)
            else:
                resp_V4, resp_V1 =get_resps(condition_type=get_get_condition_type(dataset_type), date=date, w_size=w_size, stim_off=all_ini_stim_offs[dataset_type])
                binned_epochs = get_img_resp_avg_sem(resp_V4, date, condition_type=dataset_type, get_chunks=True)  
                monkey_stats[dataset_type][date][area]['1_vs_rest_r']=get_1_vs_rest_r_monkey(binned_epochs)
                
                binned_epochs = get_img_resp_avg_sem(resp_V4, date, condition_type=dataset_type, get_chunks=True) 
                monkey_stats[dataset_type][date][area2]['1_vs_rest_r']=get_1_vs_rest_r_monkey(binned_epochs)


def get_evar_monkey_all_dates(monkey_stats, w_size=25,n_splits=10, control_shuffle=False):
    """
    Compute explained variance for all monkey dates.

    Args:
    - monkey_stats (dict): Dictionary containing monkey statistics.
    - w_size (int): Window size.

    Returns:
    - None
    """
    start_time = time.time()
    alpha_monkeys = [100, 500.0, 1000.0, 5000.0, 10000.0, 50000.0, 100000.0]
    area='V4'
    area2='V1'
    
    if control_shuffle is True:
        control_con = '_null'
    else: 
        control_con = ''
    for condition_type in monkey_stats:
        for date in monkey_stats[condition_type]:
            get_condition_type = get_get_condition_type(condition_type)
            resp_V4, resp_V1 =get_resps(condition_type=get_condition_type, date=date, 
                                        w_size=w_size,stim_off=all_ini_stim_offs[condition_type])
            alpha, evars = get_best_alpha_evars(resp_V1, resp_V4, n_splits=n_splits, alphas=alpha_monkeys,
                                                frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
            alpha2, evars2 = get_best_alpha_evars(resp_V4, resp_V1, n_splits=n_splits, 
                                                frames_reduced=all_frames_reduced[condition_type], control_shuffle=control_shuffle)
            monkey_stats[condition_type][date][area]['evars' + control_con]=evars
            monkey_stats[condition_type][date][area2]['evars' + control_con]=evars2
            monkey_stats[condition_type][date][area]['alpha' + control_con]=alpha
            monkey_stats[condition_type][date][area2]['alpha' + control_con]=alpha2
            print(date,'done')
        print(condition_type, 'done')
    end_time = time.time()
    elapsed_time = (end_time - start_time)/60
    print(f'yay! it took {elapsed_time:.2f} minutes to finish all dataset types!')
