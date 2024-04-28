import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections


main_dir = ''
func_dir = main_dir + 'utils/'

import sys
sys.path.insert(0,func_dir)
from ridge_regression_functions import get_predictions_evars_parallel
from stats_functions import get_t_test_stars, get_oneway_anova_stars


all_frames_reduced = {'SNR': 5, 'SNR_spont': 5, 'RS': 20, 
                    'RS_open':20, 'RS_closed': 20, 
                    'RF_thin':25, 'RF_large':25}
all_ini_stim_offs = {'SNR': 400, 'SNR_spont': 200, 'RS': None,
                    'RS_open':None, 'RS_closed': None, 
                    'RF_thin':1000, 'RF_large':1000, 'RF_thin_spont':200,'RF_large_spont':200}

condition_types =['SNR', 'SNR_spont', 'RS', 'RS_open', 'RS_closed', 'RF_thin', 'RF_large','RF_thin_spont','RF_large_spont']


def get_subsamples(resp_array1, resp_array2, date, frame_lengths, seed=None):
    """gets frames subsamples from the different stimulus type recordings

    Args:
        resp_array1 (_type_): recording from area 1
        resp_array2 (_type_): recording from area 2
        date (_type_): date belonging to recording activity
        frame_lengths (_type_): dictionary containing the number of frames to subsample
        seed (_type_, optional): Defaults to None.

    Returns:
        resp_array1_subsampled, resp_array2_subsamped
    """
    if seed is not None:
        np.random.seed(seed)
    start_idx = np.random.randint(0,len(resp_array1)-frame_lengths[date])
    stop_idx = start_idx + frame_lengths[date]
    return resp_array1[start_idx:stop_idx], resp_array2[start_idx:stop_idx]
    

def process_evar_subsample_seeds(input_resp, pred_resp, date, min_lengths, seed, alpha, alpha2, condition_type, n_splits=10, control_shuffle=False):
    """gets the inter-area prediction by subsampling frame size to make sure all stimulus types have a controlled training size

    Args:
        input_resp (_type_): area1 activity shaped (n_frames, n_neurons_or_sites)
        pred_resp (_type_): area2 activity shaped (n_frames, n_neurons_or_sites)
        date (_type_): date of recording
        min_lengths (_type_): _description_
        seed (_type_): _description_
        alphas (_type_): _description_
        condition_type (_type_): _description_
        n_splits (int, optional): _description_. Defaults to 10.
        control_shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: the prediction performances from both area1 and area2
    """
    pred_resp_sub, input_resp_sub= get_subsamples(pred_resp, input_resp, date, min_lengths, seed=seed)
        
    _, pred_evars = get_predictions_evars_parallel(input_resp_sub, pred_resp_sub, n_splits=n_splits, 
                                                    frames_reduced = all_frames_reduced[condition_type],
                                                    alpha=alpha, control_shuffle=control_shuffle)
    _, input_evars = get_predictions_evars_parallel(pred_resp_sub, input_resp_sub, n_splits=n_splits, 
                                                    frames_reduced = all_frames_reduced[condition_type],
                                                    alpha=alpha2, control_shuffle=control_shuffle)
    return pred_evars, input_evars


def extract_mouse_name(input_string):
    index_of_MP = input_string.find('MP')
    return input_string[index_of_MP:index_of_MP + 5] if index_of_MP != -1 and index_of_MP + 5 <= len(input_string) else None
def get_property_dataset_type(input_string):
    if 'spont' in input_string:
        return input_string.replace('_spont','')
    else:
        return input_string 
def make_mouse_df(mouse_stats_, dataset_types=['ori32','natimg32']):
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
                for split_half_r, snr,max_corr_val, evar, null_evar in zip(split_half_rs, SNRs,values['max_corr_val'],values['evars'],values['evars_null']):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Mouse': mouse,
                        'Mouse Name':mouse_name,
                        'Area': area_,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'Split-half r': split_half_r,
                        'max corr. val':max_corr_val,
                        'control_shuffle':False, 
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
                        'Split-half r': split_half_r,
                        'max corr. val':max_corr_val,
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

def make_monkey_df(monkey_stats_, dataset_types=['SNR', 'RF_thin', 'RF_large'], spont_comparisons=False):
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
                if spont_comparisons is True:
                    if 'spont' not in dataset_type or 'RS' in dataset_type:
                        evars = values['spont_comparison_evars']
                        evars_null = values['spont_comparison_evars_null']
                        if len(evars.shape)>1:
                            evars = np.mean(evars, axis=0)
                            evars_null = np.mean(evars_null, axis=0)
                    else:
                        evars = values['evars']
                        evars_null = values['evars_null']
                else:
                    evars = values['evars']
                    evars_null = values['evars_null']
                
                for split_half_r, snr,max_corr_val, evar, null_evar in zip(split_half_rs, SNRs,values['max_corr_val'],evars,evars_null):
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date':date,
                        'Area': area,
                        'Direction':direction,
                        'EV': evar,
                        'SNR': snr,
                        'max corr. val':max_corr_val,
                        'Split-half r': split_half_r,
                        'control_shuffle':False, 
                    })
                    data.append({
                        'Dataset Type': dataset_type,
                        'Activity Type': act_type,
                        'Date': date,
                        'Area': area,
                        'Direction':direction,
                        'EV': null_evar,
                        'SNR': snr,
                        'max corr. val':max_corr_val,
                        'Split-half r': split_half_r,
                        'control_shuffle':True, 
                    })
    # Create a DataFrame from the flattened data
    df_monkey_all = pd.DataFrame(data)
    return df_monkey_all


def add_stars_2_sets(df_, neuron_property, x, x_order, hue, ax, fontsize=7, height1=0.97, height2=0.97, perm_t=True, perm_type='ind', hierarchical=False, mouse_or_date='Mouse Name'):
    """
    Add significance stars to a grouped bar plot representing two sets of data.

    Parameters:
    -----------
    df_ : pandas.DataFrame
        The DataFrame containing the data.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    x : str
        The name of the column representing the x-axis variable (e.g., grouping variable).
    x_order : list
        The order of values for the x-axis variable.
    hue : str
        The name of the column representing the hue variable (e.g., subgrouping variable).
    ax : matplotlib.axes.Axes
        The Axes object on which the plot is drawn.
    fontsize : int, optional
        Font size for the significance stars (default is 7).
    height1 : float, optional
        Height for the first set of significance stars (default is 0.97).
    height2 : float, optional
        Height for the second set of significance stars (default is 0.97).
    perm_t : bool, optional
        Whether to perform permutation tests instead of traditional t-tests (default is True).
    perm_type : {'ind', 'paired'}, optional
        Type of permutation test to perform. 'ind' for independent samples permutation test, 'paired' for paired samples permutation test (default is 'ind').
    hierarchical : bool, optional
        Whether hierarchical permutation test with animal bootstrapping should be performed (default is False).
    mouse_or_date : str, optional
        The name of the column representing the mouse or date (default is 'Mouse Name').

    Notes:
    ------
    This function calculates significance stars for pairwise comparisons of two sets of data based on the provided DataFrame.
    The significance stars indicate the level of significance for each pairwise comparison: *** for p < 0.001, ** for p < 0.01, * for p < 0.05, and 'n.s.' for not significant.
    """
    stars1 = get_t_test_stars(df_[df_[x]==x_order[0]], hue, neuron_property, perm_t=perm_t, perm_type=perm_type, hierarchical=hierarchical, mouse_or_date=mouse_or_date)
    stars2 = get_t_test_stars(df_[df_[x]==x_order[1]], hue, neuron_property, perm_t=perm_t, perm_type=perm_type,hierarchical=hierarchical, mouse_or_date=mouse_or_date)
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



def add_anova_stars_2_sets_5_conditions(df_, neuron_property, x, x_order, hue, hue_order, ax, fontsize=5, height1=0.73, 
                        height2=0.82,height3=0.91, height4=1, stars1_positions=None, stars2_positions=None):
    """
    Add significance stars to a grouped bar plot representing two sets of data with five conditions.

    Parameters:
    -----------
    df_ : pandas.DataFrame
        The DataFrame containing the data.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    x : str
        The name of the column representing the x-axis variable (e.g., grouping variable).
    x_order : list
        The order of values for the x-axis variable.
    hue : str
        The name of the column representing the hue variable (e.g., subgrouping variable).
    hue_order : list
        The order of values for the hue variable.
    ax : matplotlib.axes.Axes
        The Axes object on which the plot is drawn.
    fontsize : int, optional
        Font size for the significance stars (default is 5).
    height1 : float, optional
        Height for the first set of significance stars (default is 0.73).
    height2 : float, optional
        Height for the second set of significance stars (default is 0.82).
    height3 : float, optional
        Height for the third set of significance stars (default is 0.91).
    height4 : float, optional
        Height for the fourth set of significance stars (default is 1).
    stars1_positions : list, optional
        The positions for the significance stars associated with the first set of data (default is None).
    stars2_positions : list, optional
        The positions for the significance stars associated with the second set of data (default is None).

    Notes:
    ------
    This function calculates significance stars for pairwise comparisons of two sets of data with five conditions based on the provided DataFrame.
    The significance stars indicate the level of significance for each pairwise comparison: *** for p < 0.001, ** for p < 0.01, * for p < 0.05, and 'n.s.' for not significant.
    """
    labels1,all_stars1 = get_oneway_anova_stars(df_[df_[x]==x_order[0]], hue, hue_order, neuron_property, perm_t=True, perm_type='paired')
    _, all_stars2 = get_oneway_anova_stars(df_[df_[x]==x_order[1]], hue, hue_order, neuron_property,perm_t=True, perm_type='paired')
    height_positions1 = [height1,height2,height3,height4]
    height_positions2 = [height1,height2,height3,height4]
    if stars1_positions is None:
        stars1_positions= [0.14,0.19,0.23,0.27]
    if stars2_positions is None:
        stars2_positions= [0.65, 0.7,0.74,0.78]
    bar_halflength = 0.06
    offset_=0
    for s, (star1, star2) in enumerate(zip(all_stars1[:4], all_stars2[:4])):
        if s>0:
            ax.hlines(height_positions1[s]- 0.01, stars1_positions[s] - bar_halflength-offset_, stars1_positions[s] + bar_halflength+offset_, 
                        color='black',transform=ax.transAxes, linewidth=0.5)
            ax.hlines(height_positions2[s]- 0.01, stars2_positions[s] - bar_halflength-offset_, stars2_positions[s] + bar_halflength+offset_, 
                        color='black',transform=ax.transAxes, linewidth=0.5)
        if star1 =='n.s.':
            height_positions1[s]=height_positions1[s]+0.02
            # color1='#C0C0C0'
            color1='black'
        else:
            color1='black'
        if star2 =='n.s.':
            height_positions2[s]=height_positions2[s]+0.02
            # color2='#C0C0C0'
            color2='black'
        else:
            color2='black'
        ax.text(stars1_positions[s], height_positions1[s], star1, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes,color=color1)
        ax.text(stars2_positions[s], height_positions2[s], star2, ha='center', va='center', fontsize=fontsize, transform=ax.transAxes, color=color2)
        
        offset_+=0.04


def fig5_violinplot(df, x, y, hue, ax, y_label, linewidth=0.9,
                    palette=['gray','lightgray'], fontsize=7,plot_control_ev=True, 
                    animal='mouse',show_legend=False,leg_loc=(1,0.3),**args):
    if animal == 'mouse':
        label_order=['L4→L2/3', 'L2/3→L4']
        hue_order=['stimulus', 'gray screen']
    elif animal =='monkey':
        label_order=['V1→V4', 'V4→V1']
        hue_order=['stimulus', 'gray screen','lights off',
                'lights off\neyes open','lights off\neyes closed']
        
    sns.violinplot(x=x, y=y, hue=hue, 
                data=df[df['control_shuffle']==False],ax=ax,order=label_order, hue_order=hue_order,
                inner='box',linewidth=linewidth,saturation=1,
                inner_kws={'box_width':2, 'whis_width':0.5,
                            'marker':'_', 'markersize':3,
                            'markeredgewidth':0.8,
                            },palette=palette,
                            **args
                            )
    sns.despine()
    ax.set_ylabel(y_label, fontsize=fontsize, labelpad=1)
    ax.set_xlabel(None)
    ax.legend_.remove()
    ax.spines[:].set_linewidth(0.3)
    if plot_control_ev is True:
        data = df[df['control_shuffle']==True][y]
        per_25 = np.percentile(data.dropna().values, 25)
        per_75 = np.percentile(data.dropna().values, 75)
        ax.axhspan(per_25, per_75, alpha=0.1, color='blue', label='shuffle\ncontrol IQR',
                linewidth=0,
                )
    if animal =='mouse':
        add_stars_2_sets(df[df['control_shuffle']==False], y, x, label_order, hue, ax, 
                    hierarchical=True, height1=0.7,height2=0.7, perm_type='paired')
    else:
        add_anova_stars_2_sets_5_conditions(df[df['control_shuffle']==False], y, x, label_order, hue, 
                                            hue_order, ax, fontsize=5)
        if show_legend is True:
            hatch_size=3
            hatches = [None,None,None,'/','.',None]
            new_handles = []
            for handle, hatch in zip(ax.get_legend_handles_labels()[0], hatches):
                if hatch is not None:
                    handle.set_hatch(hatch*hatch_size)
                new_handles.append(handle)
            ax.legend(loc=leg_loc, handles = new_handles, fontsize=6)
    
    ax.tick_params(axis='both', labelsize=fontsize, width=0.5, length=3, pad=2)
    add_custom_colors(ax, animal=animal)
    

def add_custom_colors(ax, animal='mouse'):
    if animal=='mouse':
        custom_colors = ['#72BEB7','#B6E3DF','#EDAEAE', '#f6d6d6']
        violins  = [s for s in ax.get_children() if isinstance(s, collections.PolyCollection)]
        for violin, color in zip(violins, custom_colors):
            violin.set_facecolor(color)
    else:
        custom_colors = ['#72BEB7','#B6E3DF','#136a66','#136a66','#136a66',
                '#EDAEAE', '#f6d6d6','#a85959','#a85959','#a85959',]
        violins  = [s for s in ax.get_children() if isinstance(s, collections.PolyCollection)]
        for violin, color in zip(violins, custom_colors):
            violin.set_facecolor(color)
        inners = ax.get_lines()
        violin_indices= [2,3,4,7,8,9]
        inner_first_indices = [6,9,12,21,24,27]
        ofsett_val=0.06
        hatch_size=3
        for v,i in zip(violin_indices, inner_first_indices):
            violins[v].get_paths()[0].vertices[:,0] += ofsett_val
            inners[i].get_path().vertices[:,0]+= ofsett_val 
            inners[i+1].get_path().vertices[:,0]+= ofsett_val
            inners[i+2].get_path().vertices[:,0]+= ofsett_val
            if v==3 or v==8:
                violins[v].set_hatch('/'*hatch_size)
            if v==4 or v==9:
                violins[v].set_hatch('.'*hatch_size)
        

def plot_corrs_sns(corr_df, x, y, ax, xy=(0.05, 0.5), area='L2/3', color='black', 
                rcolor='black', alpha=0.5, s=8,fontsize=7, xlim=(-0.03,0.9), ylim=(-0.03,0.9), 
                tick_values = [0, 0.4, 0.8, ],**kwargs):
    """
    Plot scatterplot of correlations between two variables along with Pearson correlation coefficient.

    Parameters:
    -----------
    corr_df : pandas.DataFrame
        DataFrame containing correlation data.
    x : str
        Name of the column representing the x-axis variable.
    y : str
        Name of the column representing the y-axis variable.
    ax : matplotlib.axes.Axes
        Axes object on which the plot is drawn.
    xy : tuple, optional
        Tuple containing the position for the annotation of the Pearson correlation coefficient (default is (0.05, 0.5)).
    area : str, optional
        Name of the brain area to plot correlations for (default is 'L2/3').
    color : str, optional
        Color for the scatterplot markers (default is 'black').
    rcolor : str, optional
        Color for the annotation text of the Pearson correlation coefficient (default is 'black').
    alpha : float, optional
        Transparency level of the scatterplot markers (default is 0.5).
    s : int, optional
        Size of the scatterplot markers (default is 8).
    fontsize : int, optional
        Font size for text elements (default is 7).
    xlim : tuple, optional
        Limits for the x-axis (default is (-0.03,0.9)).
    ylim : tuple, optional
        Limits for the y-axis (default is (-0.03,0.9)).
    tick_values : list, optional
        List of tick values for the x and y axes (default is [0, 0.4, 0.8]).
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the scatterplot function.

    Notes:
    ------
    This function plots a scatterplot of correlations between two variables specified by 'x' and 'y' columns of the provided DataFrame.
    The 'area' parameter allows focusing on correlations within a specific brain area.
    Pearson correlation coefficient is calculated and annotated on the plot.
    """
    df_area= corr_df[corr_df['Area']==area]
    if area =='L2/3' or area=='V4':
        color='lightseagreen'
        line_color='darkcyan'
    elif area=='L4' or area=='V1':
        color='lightcoral'
        line_color='#a85959'
    h = sns.scatterplot(x=x, y=y, color= color, 
                    data=df_area, alpha=alpha,s=s,ax=ax
            )
    sns.despine()
    
    h.tick_params(labelsize=16)
    pearson_corr = df_area[x].corr(df_area[y], method='pearson')
    h.annotate(f'r={pearson_corr:.2f}', xy=xy, fontsize=fontsize, color=rcolor)
    ax.set(xlim=xlim,
        ylim=ylim)
    tick_values = tick_values  # Define the tick values
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    ax.tick_params(axis='both', labelsize=fontsize, width=0.5, length=2, pad=2)
    ax.set_xlabel('EV fraction\nstimulus',fontsize=fontsize, labelpad=1)
    ax.set_ylabel('EV fraction\ngray screen', fontsize=fontsize, labelpad=1)
    ax.spines[:].set_linewidth(0.3)
    max_val = df_area[x].max()
    ax.plot([0,max_val+0.05],[0,max_val + 0.05], color=line_color, linestyle='--', linewidth=1,)
    
##supplemental
    

def fig5_supp_violinplot(df,area, x, y, hue, ax, y_label, linewidth=0.9,
                    palette=['gray','lightgray'], fontsize=7,plot_control_ev=True, 
                    show_legend=False,leg_loc=(1,0.3),stars_height=0.7,**args):
    """
    Create a violin plot for supplementary figure 5.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    area : str
        The brain area for which the plot is created.
    x : str
        The column name representing the x-axis variable.
    y : str
        The column name representing the y-axis variable.
    hue : str
        The column name representing the hue variable.
    ax : matplotlib.axes.Axes
        The axes object on which the plot will be drawn.
    y_label : str
        Label for the y-axis.
    linewidth : float, optional
        Width of the violin plot lines (default is 0.9).
    palette : list of str, optional
        Color palette for the violin plot (default is ['gray','lightgray']).
    fontsize : int, optional
        Font size for text elements (default is 7).
    plot_control_ev : bool, optional
        Whether to plot the control interquartile range (default is True).
    show_legend : bool, optional
        Whether to show the legend (default is False).
    leg_loc : tuple, optional
        Location of the legend (default is (1,0.3)).
    stars_height : float, optional
        Height for displaying significance stars (default is 0.7).
    **args : dict, optional
        Additional keyword arguments to be passed to the seaborn.violinplot function.

    Notes:
    ------
    This function creates a violin plot for supplementary figure 5. It visualizes the distribution of the 'y' variable 
    across different levels of the 'x' variable, with the distribution split by the 'hue' variable. The 'area' parameter
    specifies the brain area for which the plot is generated. Significance stars indicating the result of paired t-tests
    are displayed on top of each group in the plot.
    """
    if area=='L2/3':
        palette=['#72BEB7','#B6E3DF']
    elif area=='L4':
        palette=['#EDAEAE', '#f6d6d6']
    label_order=['MP031', 'MP032','MP033']
    hue_order=['stimulus', 'gray screen']
        
    sns.violinplot(x=x, y=y, hue=hue, 
                data=df[(df['control_shuffle']==False)&(df.Area==area)],ax=ax,order=label_order, hue_order=hue_order,
                inner='box',linewidth=linewidth,saturation=1,
                inner_kws={'box_width':2, 'whis_width':0.5,
                            'marker':'_', 'markersize':3,
                            'markeredgewidth':0.8,
                            },palette=palette,
                            **args
                            )
    sns.despine()
    ax.set_ylabel(y_label, fontsize=fontsize, labelpad=1)
    ax.set_xlabel(None)
    
    ax.spines[:].set_linewidth(0.3)
    if plot_control_ev is True:
        data = df[df['control_shuffle']==True][y]
        per_25 = np.percentile(data.dropna().values, 25)
        per_75 = np.percentile(data.dropna().values, 75)
        ax.axhspan(per_25, per_75, alpha=0.1, color='blue', label='shuffle\ncontrol IQR',
                linewidth=0,
                )
    x_positions = [0.18,0.5,0.85]
    for m, mouse in enumerate(label_order):
        star_fontsize=fontsize*.9
        star = get_t_test_stars(df[(df['control_shuffle']==False)&(df.Area==area)&(df[x]==mouse)], hue, y, perm_t=True, perm_type='paired', hierarchical=False)
        if star == 'n.s.':
            stars_height = stars_height + 0.015
            star_fontsize = star_fontsize*0.9
        ax.text(x_positions[m], stars_height, star, ha='center', va='center', fontsize=star_fontsize, transform=ax.transAxes)    
        ax.tick_params(axis='both', labelsize=fontsize, width=0.5, length=3, pad=2)
        
    if show_legend is False:
        ax.legend_.remove()
    else:
        ax.legend(loc=leg_loc)
    