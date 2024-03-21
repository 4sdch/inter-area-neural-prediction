main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/inter_areal_predictability/'
func_dir = main_dir + 'utils/'

import sys
sys.path.insert(0,func_dir)

main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/'

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension, _PREFIXES_FACTORS, _LATEX_MU

class TimeDimension(_Dimension):
    def __init__(self):
        super().__init__("s")
        for prefix, factor in _PREFIXES_FACTORS.items():
            latexrepr = None
            if prefix == "\u00b5" or prefix == "u":
                latexrepr = _LATEX_MU + "s"
            self.add_units(prefix + "s", factor, latexrepr)
            

def plot_correlations(df_, color, r_position, ax):
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
                ax.set_xticks(ticks=np.arange(16*frame_start, 16*frame_stop, 16*2), 
                              labels=np.arange(frame_start, frame_stop, 2))
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