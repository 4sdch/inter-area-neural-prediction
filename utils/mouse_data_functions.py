import scipy.io as sio
from io import BytesIO
import numpy as np
from scipy.sparse.linalg import eigsh
import copy

main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/inter_areal_predictability/'

class mt_retriever:
    ori32_filenames = ['ori32_M160825_MP027_2016-12-15','ori32_M170604_MP031_2017-06-26', 'ori32_M170714_MP032_2017-08-02', 'ori32_M170717_MP033_2017-08-17']
    natimg32_filenames = ['natimg32_M170604_MP031_2017-06-27','natimg32_M170714_MP032_2017-08-01','natimg32_M170717_MP033_2017-08-25']
    # omitting natimg32_M150824_MP019_2016-03-23 due to no cells in indices 385 and 420
    def __init__(self, main_dir, dataset_type):
        ## loads all the info we need 
        """
        Initialize mt_retriever object.

        Parameters:
        - main_dir: Main directory path
        - dataset_type: Type of dataset ('ori32' or 'natimg32')
        """
        self.mts = {}
        if 'ori32' in dataset_type:
            self.filenames = copy.deepcopy(self.ori32_filenames)
        elif 'natimg32' in dataset_type:
            self.filenames = copy.deepcopy(self.natimg32_filenames)
        for dataset in self.filenames:
            with open(main_dir + f'data/stringer/{dataset}', mode='rb') as file: # b is important -> binary
                fileContent = file.read()
                self.mts[dataset] = sio.loadmat(BytesIO(fileContent))
            
    def set_raw_responses(self):
        """
        Set raw responses for stimulus and spontaneous activity.

        Returns:
        - resp: Stimulus responses
        - istim: Identifiers of stimuli in resp
        - spont: Spontaneous activity
        """
        #Giorgia OG code
        ### stimulus responses
        resp = self.mt['stim'][0]['resp'][0]    # stimuli by neurons  ## stimulus response by neurons?
        istim = copy.deepcopy(self.mt['stim'][0]['istim'][0])  # identities of stimuli in resp
        spont = self.mt['stim'][0]['spont'][0]  # timepoints by neurons
        return resp, istim, spont

    def add_preprocessing(self):
        """
        Preprocess raw responses.

        Returns:
        - resp: Preprocessed responses
        - spont: Preprocessed spontaneous activity
        """
        resp, istim, spont = self.set_raw_responses()
        #### Now add preprocessing --> from notebook https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/powerlaws.ipynb
        istim -= 1 # get out of MATLAB convention
        istim = istim[:,0]
        nimg = istim.max() # these are blank stims (exclude them) ## it looks like there are 260 of them that have the max val of 2801
        resp = resp[istim<nimg, :]
        self.istim = istim[istim<nimg] ##not sure why this was done after the indexing
        return resp, spont
    
    def subtract_spont(self):
        """
        Subtract spontaneous activity and normalize responses.

        Returns:
        - resp: Normalized responses
        - spont: Normalized spontaneous activity
        """
        resp, spont = self.add_preprocessing()
        mu = spont.mean(axis=0)
        sd = spont.std(axis=0) + 1e-6
        resp = (resp - mu) / sd
        spont = (spont - mu) / sd

        return resp, spont
    
    def subtract_spot_and_mean_center(self):
        """
        Subtract spontaneous activity, mean center, and normalize responses.

        Returns:
        - resp: Normalized and mean-centered responses
        - spont: Normalized spontaneous activity
        """
        resp, spont = self.add_preprocessing()
        mu = spont.mean(axis=0)
        sd = spont.std(axis=0) + 1e-6
        resp = (resp - mu) / sd
        spont = (spont - mu) / sd
        # mean center each neuron ### important for PCA 
        resp-= resp.mean(axis=0)
        return resp, spont
    
    def subtract_spont_pc_and_mean_center(self):
        """
        Subtract spontaneous activity, project onto principal components, and mean center responses.

        Returns:
        - resp: Mean-centered responses
        - spont: Normalized spontaneous activity
        """
        resp, spont = self.add_preprocessing()
        # subtract spont (32D)  ## i have to go over this
        mu = spont.mean(axis=0)
        sd = spont.std(axis=0) + 1e-6
        resp = (resp - mu) / sd
        spont = (spont - mu) / sd
        sv,u = eigsh(spont.T @ spont, k=32)
        resp = resp - (resp @ u) @ u.T            
        # mean center each neuron ### important for PCA 
        resp -= resp.mean(axis=0)

        return resp, spont
    
    def subtract_spont_pc(self):
        """
        Subtract spontaneous activity and project onto principal components.

        Returns:
        - resp: Responses after PCA
        - spont: Normalized spontaneous activity
        """
        resp, spont = self.add_preprocessing()
        # subtract spont (32D)  ## i have to go over this
        mu = spont.mean(axis=0)
        sd = spont.std(axis=0) + 1e-6
        resp = (resp - mu) / sd
        spont = (spont - mu) / sd
        sv,u = eigsh(spont.T @ spont, k=32)
        resp = resp - (resp @ u) @ u.T            
        return resp, spont


    def get_L_indices(self):
        """
        Get indices of neurons in each layer.

        Returns:
        - L1indices: Indices of neurons in Layer 1
        - L23indices: Indices of neurons in Layer 2/3
        -
        """
        resp, spont = self.subtract_spot_and_mean_center()
        n_neurons = np.shape(resp)[1]
        
        # get cell xyz data
        med = self.mt['med']
        
        # extract the coordinates x,y,z of the neurons 
        medT = med.T # transpose to have vectors per x/y/z dimension instead of vector per neuron
        zs = medT[2,:]
        
        # indices for each area
        n_all = np.arange(n_neurons)

        #new indices per Giorgia 2_14
        L2min=125 
        L2max= 225
        L3min=225
        L3max= 325
        L4min=350
        L4max=450 
        
        L1indices = [int(n) for n in n_all if zs[n] <= L2min]
        L2indices = [int(n) for n in n_all if zs[n] <= L2max and zs[n] >L2min]
        L3indices = [int(n) for n in n_all if zs[n] <= L3max and zs[n] >L3min]
        L23indices = [int(n) for n in n_all if zs[n] <= L3max and zs[n] >L2min]
        L4indices = [int(n) for n in n_all if zs[n] <= L4max and zs[n] >L4min]

        return L1indices, L23indices, L2indices, L3indices, L4indices
    
    def retrieve_layer_activity(self, activity_type, dataset_name, mean_center=False, removed_pc = False):
        """
        Retrieve layer activity based on specified activity type and dataset.

        Parameters:
        - activity_type: Type of activity ('resp' for responses, 'spont' for spontaneous activity)
        - dataset_name: Name of the dataset to retrieve activity from
        - mean_center: Boolean indicating whether to mean-center the responses (default: False)
        - removed_pc: Boolean indicating whether to remove principal components from the responses (default: False)

        Returns:
        - If activity_type is 'resp':
            - resp_L1: Responses for Layer 1
            - resp_L23: Responses for Layer 2/3
            - resp_L2: Responses for Layer 2
            - resp_L3: Responses for Layer 3
            - resp_L4: Responses for Layer 4
        - If activity_type is 'spont':
            - spont_L1: Spontaneous activity for Layer 1
            - spont_L23: Spontaneous activity for Layer 2/3
            - spont_L2: Spontaneous activity for Layer 2
            - spont_L3: Spontaneous activity for Layer 3
            - spont_L4: Spontaneous activity for Layer 4
        """
        self.mt = self.mts[dataset_name]
        if mean_center is True:
            resp, spont = self.subtract_spot_and_mean_center()
        elif removed_pc is True:
            resp, spont = self.subtract_spont_pc()
        else:
            resp, spont = self.subtract_spont()
        L1indices, L23indices, L2indices, L3indices, L4indices = self.get_L_indices()
        
        resp_L1 = resp[:,L1indices]
        resp_L23 = resp[:,L23indices]
        resp_L2 = resp[:,L2indices]
        resp_L3 = resp[:,L3indices]
        resp_L4 = resp[:,L4indices]
        
        spont_L1 = spont[:,L1indices]
        spont_L23 = spont[:,L23indices]
        spont_L2 = spont[:,L2indices]
        spont_L3 = spont[:,L3indices]
        spont_L4 = spont[:,L4indices]
        
        if 'resp' in activity_type:
            # extract responses for all areas
            return resp_L1, resp_L23, resp_L2, resp_L3, resp_L4
        elif 'spont' in activity_type:
            return spont_L1, spont_L23, spont_L2, spont_L3, spont_L4