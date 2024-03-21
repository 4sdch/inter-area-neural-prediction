
from neo import NixIO
import pandas as pd
import numpy as np
import copy

main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/inter_areal_predictability/'


def get_epoch_times(resp_array, date, stim_on=0, stim_off=400, spont_stim_off =200, monkey='L'):
    """
    This function reads metadata, processes epoch times, and extracts response and spontaneous activity
    arrays based on specified parameters.
    
    :param resp_array: The `resp_array` parameter in the `get_epoch_times` function is likely a numpy
    array containing neural responses data. This array is used to extract specific epochs of neural
    responses based on the provided parameters and return them as `true_resp` and `true_spont`
    :param date: The `date` parameter is used to specify the date for which you want to retrieve epoch
    times
    :param stim_on: The `stim_on` parameter in the `get_epoch_times` function represents the time (in
    milliseconds) when the stimulus starts in each epoch. By default, it is set to 0, but you can
    customize it if needed, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_epoch_times` function represents the time in
    milliseconds when the stimulus ends during data processing. It is used to calculate the duration of
    the response window for each epoch, defaults to 400 (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_epoch_times` function represents
    the duration of spontaneous activity after the original stimulus onset. It is used to extract the
    spontaneous responses from the `resp_array` data. In the function, it is subtracted from the
    original stimulus onset time to, defaults to 200 (optional)
    :param monkey: The `monkey` parameter in the `get_epoch_times` function is used to specify which
    monkey's data to retrieve. It is a string parameter that indicates the monkey's name or identifier,
    defaults to L (optional)
    :return: The function `get_epoch_times` returns two arrays: `true_resp` and `true_spont`.
    `true_resp` contains responses data for each epoch with a specified stimulus duration, while
    `true_spont` contains spontaneous responses data for each epoch.
    """
    # get epoch times and convert to ms frames.
    df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_SNR_{date}.csv')
    new_df = df[df['success']].copy()
    new_df['og_stim_on'] = (new_df['t_stim_on'] *1000).round().astype(int)
    new_df['new_stim_on'] = (new_df['t_stim_on'] *1000).round().astype(int) + stim_on
    # separate responses and make into a (epochs,frames_per_epoch, n_electrodes) array.
    # To make it more uniform, made all the epoch times last 400ms (ranges from 400ms to 410ms per epoch)

    resp_times = stim_off-stim_on
    true_resp = resp_array[new_df['new_stim_on'].values[:, None] + np.arange(resp_times), :]
    true_spont = resp_array[(new_df['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    
    return true_resp, true_spont

def get_epoch_times_RF(resp_array, date, stim_on=0, stim_off=1000, spont_stim_off =200, monkey='L', direction_n=None):
    """
    This function reads metadata from a CSV file, processes the data, and returns specific arrays based
    on input parameters for a moving bars dataset.
    
    :param resp_array: The `resp_array` parameter in the `get_epoch_times_RF` function is expected to be
    a NumPy array containing response data. This array likely represents neural responses to stimuli in
    a neuroscience experiment. The function processes this response data based on the other parameters
    provided to extract specific epochs of interest related to
    :param date: The `date` parameter in the `get_epoch_times_RF` function is used to specify the date
    for which you want to retrieve epoch times and convert them to milliseconds frames for the moving
    bars dataset
    :param stim_on: The `stim_on` parameter in the `get_epoch_times_RF` function represents the time (in
    milliseconds) when the stimulus starts during the experiment. It is used to calculate the new
    stimulus onset time based on the original stimulus onset time in the dataset, defaults to 0
    (optional)
    :param stim_off: The `stim_off` parameter in the `get_epoch_times_RF` function represents the time
    in milliseconds when the stimulus ends during data processing for the moving bars dataset. It is
    used to calculate the duration of the response window after the stimulus onset (`stim_on`) and to
    extract the corresponding response data from, defaults to 1000 (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_epoch_times_RF` function
    represents the duration in milliseconds after the original stimulus onset time (`og_stim_on`) for
    which you want to capture responses for the spontaneous activity. It is used to define the time
    window for extracting responses before the, defaults to 200 (optional)
    :param monkey: The `monkey` parameter in the `get_epoch_times_RF` function is used to specify the
    monkey for which the data is being processed. It is a string parameter that indicates the name of
    the monkey (e.g., 'L' for monkey L). This parameter is used to load the corresponding metadata,
    defaults to L (optional)
    :param direction_n: The `direction_n` parameter specifies whether to retrieve only one direction of
    sweeping bar. The options are integers from 0 to 3. If you provide a value for `direction_n`, the
    function will filter the data to retrieve responses only for that specific direction of sweeping bar
    :return: The function `get_epoch_times_RF` returns three values: `true_resp`, `true_spont`, and
    `cond_labels`.
    """
    # get epoch times and convert to ms frames for moving bars dataset.
    df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_RF_{date}.csv')
    new_df = df[df['success']].copy()
    new_df['og_stim_on'] = (new_df['t_stim_on'] *1000).round().astype(int)
    new_df['new_stim_on'] = (new_df['t_stim_on'] *1000).round().astype(int) + stim_on
    
    mapping = {}
    for count, label in enumerate(new_df['cond'].unique()):
        mapping[label]=count
    mapping
    new_df['cond_num']=new_df['cond'].map(mapping)

    resp_times = stim_off-stim_on

    if direction_n is not None:
        ## specifies whether to retrieve only one direction of sweeping bar. options are 0 to 3.
        query = f'cond_num == {direction_n}'
        true_resp = resp_array[new_df.query(query)['new_stim_on'].values[:, None] + np.arange(resp_times), :]
        true_spont = resp_array[(new_df.query(query)['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    else:
        true_resp = resp_array[new_df['new_stim_on'].values[:, None] + np.arange(resp_times), :]
        true_spont = resp_array[(new_df['og_stim_on'] - int(spont_stim_off)).values[:, None] + np.arange(spont_stim_off), :]
    
    labels = new_df.cond_num.values
    cond_labels = np.tile(labels,(1000,1)).T
    cond_labels= cond_labels[:,:,np.newaxis]

    return true_resp, true_spont, cond_labels


def get_number_of_epochs(date, monkey='L', condition_type='SNR'):
    """   This function retrieves the number of epochs for a specific date, monkey, and condition type from a
    CSV file.

    Args:
        date (_type_): The `date` parameter in the `get_number_of_epochs` function is used to specify the date
    for which you want to retrieve the number of epochs
        monkey (str, optional): The `monkey` parameter in the `get_number_of_epochs` function is used to specify the
    monkey for which you want to retrieve the number of epochs. By default, it is set to 'L', defaults
    to L (optional). Defaults to 'L'.
        condition_type (str, optional): It is a string parameter that specifies the condition type, such as 'SNR'
    (Signal-to-Noise Ratio) in this case, defaults to SNR (optional)
    :return: the number of epochs that meet the 'success' condition in the provided DataFrame after
    reading and filtering the data from a CSV file. Defaults to 'SNR'.

    Returns:
        _type_: _description_
    """
    # get epoch times and convert to ms frames.
    df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_{condition_type}_{date}.csv')
    new_df = df[df['success']].copy()
    return len(new_df)


def bin_labels(data, window_size, **kwargs):
    """
    The function `bin_labels` takes a dataset and aggregates the data points into bins of a specified
    window size.
    
    :param data: Data is the input array containing the datapoints that need to be binned. It is a 2D
    numpy array where each row represents a datapoint and each column represents a feature or dimension
    of the datapoint
    :param window_size: The `window_size` parameter in the `bin_labels` function represents the desired
    size of each window for binning the data points. This value determines how many data points will be
    grouped together and processed at a time during the binning process
    :return: The function `bin_labels` returns binned data where the datapoints from the input data are
    aggregated into the desired window size using the median function. The binned data is returned as a
    numpy array with dimensions [bin_datapoints, number of columns in the input data].
    """
    ## bings the datapoints from 1ms to desired window size.

    bin_datapoints = int(np.ceil(len(data)/window_size))
    binned_data = np.zeros([bin_datapoints, data.shape[1]])
    for i in range(bin_datapoints):
        window_data = data[i*window_size : (i*window_size) + window_size, :]
        binned_data[i] = np.median(window_data, axis=0)  # You can use a different aggregation function here
    return binned_data

def isolate_norm_resps_RF(resp_array, date='260617', monkey='L', bin_function=None, stim_on=0, stim_off=1000, raw_resp=False, **kwargs):
    """
    This function isolates responses for moving bars by removing gray screen presentation activity and
    optionally normalizes the responses.
    
    :param resp_array: The `resp_array` parameter is a numpy array containing neural responses
    data. This function processes this data to isolate responses for moving bars by removing gray
    screen presentation activity. The function also allows for binning the responses using a specified
    function, normalizing the responses, and returning the normalized responses
    :param date: The `date` parameter in the `isolate_norm_resps_RF` function is used to specify the
    date for which the responses are being analyzed. It has a default value of '260617' if not provided
    explicitly, defaults to 260617 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_resps_RF` function is used to specify the
    monkey from which the responses were recorded. It is a string parameter that indicates the monkey's
    identity, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_resps_RF` function is a
    function that can be applied to each epoch response in order to bin or process the data in a
    specific way. This function should take an epoch response as input and return the processed or
    binned version of that
    :param stim_on: The `stim_on` parameter in the `isolate_norm_resps_RF` function specifies the time
    point at which the stimulus starts during the experiment. It is used to determine the beginning of
    the stimulus presentation period for isolating responses to moving bars in the `resp_array`,
    defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_resps_RF` function specifies the time
    point at which the stimulus ends during data processing. It is used to define the duration of the
    stimulus presentation window. In the provided function, the default value for `stim_off` is set to,
    defaults to 1000 (optional)
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_resps_RF` function is a boolean flag
    that determines whether to return the raw responses or the normalized responses. If `raw_resp` is
    set to `True`, the function will return the true responses without normalization. If `raw_resp`,
    defaults to False (optional)
    :return: If the `raw_resp` parameter is `True`, the function will return the `true_resp` array and
    `cond_labels`. If `raw_resp` is not `True`, the function will return the `norm_resp` array and
    `binned_labels`.
    """
    ### removes the gray screen presentation activity to only obtain isolated responses for moving bars 
    true_resp, true_spont, cond_labels = get_epoch_times_RF(resp_array, stim_on=stim_on, stim_off=stim_off, date=date, monkey=monkey)

    if bin_function is not None:
        binned_resp = np.stack([bin_function(epoch_resp, **kwargs) for epoch_resp in true_resp])
        binned_spont = np.stack([bin_function(epoch_spont,**kwargs) for epoch_spont in true_spont])
        binned_labels = np.stack([bin_labels(epoch_label, **kwargs) for epoch_label in cond_labels])
    else:
        binned_resp = true_resp
        binned_spont = true_spont
        binned_labels=cond_labels

    

    if raw_resp is True:
        #return binned_resp.reshape(-1, resp_array.shape[1]), binned_labels
        #print('true resp shape', true_resp.shape)
        return true_resp, cond_labels
    else:
        norm_resp = binned_resp - np.mean(binned_spont, axis=1, keepdims=True)  
        norm_resp = norm_resp.reshape(-1, resp_array.shape[1])
        # norm_resp -= np.mean(norm_resp, axis=0)
        
        # print(norm_resp.shape)
        return norm_resp, binned_labels

def isolate_norm_spont_RF(resp_array, date='260617', monkey='L', bin_function=None, stim_on=0, stim_off=1000,raw_resp=False,spont_stim_off=300, **kwargs):
    """
    This Python function isolates responses for gray screen activity by removing moving bars activity
    and optionally binning the data.
    
    :param resp_array: `resp_array` is likely a numpy array containing responses from a neural recording
    experiment. The function `isolate_norm_spont_RF` processes this array to isolate responses to gray
    screen activity by removing moving bars activity
    :param date: The `date` parameter in the `isolate_norm_spont_RF` function is used to specify the
    date for which the responses are being analyzed. It has a default value of '260617' if not provided
    explicitly when calling the function, defaults to 260617 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_spont_RF` function is used to specify the
    monkey from which the response data is obtained, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_spont_RF` function is used to
    specify a function that will be applied to each epoch of spontaneous activity data before further
    processing. This function can be used to perform operations such as binning, averaging, or any other
    data transformation on the
    :param stim_on: The `stim_on` parameter in the `isolate_norm_spont_RF` function specifies the time
    point at which the stimulus starts during the recording. It is used to define the beginning of the
    stimulus presentation window for isolating responses related to the gray screen activity, defaults
    to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_spont_RF` function represents the
    time point at which the stimulus ends during the experiment. It is used to define the duration of
    the stimulus presentation period. In this function, it is set to a default value of 1000, defaults
    to 1000 (optional)
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_spont_RF` function is a boolean flag
    that determines whether to return the raw spontaneous responses or not. If `raw_resp` is set to
    `True`, the function will return the raw spontaneous responses along with the condition labels. If
    `, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `isolate_norm_spont_RF` function
    represents the duration (in milliseconds) after the stimulus offset where spontaneous activity is
    considered. In this function, it is used to determine the time window for isolating spontaneous
    responses after the stimulus presentation has ended, defaults to 300 (optional)
    :return: The function `isolate_norm_spont_RF` returns the normalized spontaneous responses and
    condition labels after isolating the gray screen activity from the input response array.
    """
    ### removes the moving bars activity to only obtain isolated responses for gray screen activity 
    true_resp, true_spont, cond_labels = get_epoch_times_RF(resp_array, stim_on=stim_on, stim_off=stim_off, date=date, monkey=monkey, spont_stim_off=spont_stim_off)
    if raw_resp is True:
        return true_spont, cond_labels
    if bin_function is not None:
        binned_spont = np.stack([bin_function(epoch_spont,**kwargs) for epoch_spont in true_spont])
        binned_labels = np.stack([bin_labels(epoch_label, **kwargs) for epoch_label in cond_labels])
    else:
        binned_spont = true_spont
        binned_labels=cond_labels
        
    norm_spont = binned_spont.reshape(-1, resp_array.shape[1])

    return norm_spont, binned_labels

def isolate_norm_resps(resp_array, date='250717', monkey='L', bin_function=None, stim_on=0, stim_off=400, shuffle=False, seed=None, raw_resp=False, **kwargs):
    """
    This function isolates responses for checkerboard presentations by removing gray screen activity and
    optionally performs trial shuffling or normalization.
    
    :param resp_array: `resp_array` is an array containing neural responses data, typically in the form
    of responses to checkerboard presentations
    :param date: The `date` parameter in the `isolate_norm_resps` function is used to specify the date
    for which the responses are being analyzed. It is set to a default value of '250717' but can be
    changed to any specific date for which you want to isolate responses, defaults to 250717 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_resps` function is used to specify the
    monkey from which the responses were recorded, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_resps` function allows you to
    specify a function that will be applied to each epoch response before further processing. This can
    be useful for binning or aggregating the data in a specific way before normalization or analysis
    :param stim_on: The `stim_on` parameter in the `isolate_norm_resps` function specifies the time
    point at which the stimulus presentation begins. It is used to define the start of the epoch for
    extracting responses to the checkerboard presentations, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `isolate_norm_resps` function represents the end
    time of the stimulus presentation in milliseconds. It is used to define the duration of the stimulus
    presentation window for isolating responses to checkerboard presentations, defaults to 400
    (optional)
    :param shuffle: The `shuffle` parameter in the `isolate_norm_resps` function allows you to specify
    whether you want to perform trial shuffling of checkerboard images before processing the responses.
    If `shuffle` is set to `True`, the function will shuffle the order of the responses using random
    indices before further, defaults to False (optional)
    :param seed: The `seed` parameter in the `isolate_norm_resps` function is used to set the random
    seed for reproducibility when shuffling the indices of the responses. By setting a specific seed
    value, you can ensure that the same random shuffling is applied each time the function is called
    with
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_resps` function is a boolean flag
    that determines whether to return the raw responses without normalization or not. If `raw_resp` is
    set to `True`, the function will return the binned responses reshaped as a 2D array, defaults to
    False (optional)
    :return: The function `isolate_norm_resps` returns either the reshaped binned responses or the
    normalized responses based on the input parameters. If `raw_resp` is True, it returns the binned
    responses reshaped as a 2D array. If `raw_resp` is False, it returns the normalized responses
    calculated as the binned responses minus the mean of the binned spontaneous responses, resh
    """
    ### removes the gray screen presentation activity to only obtain isolated responses for checkerboard presentations
    true_resp, true_spont = get_epoch_times(resp_array, stim_on=stim_on, stim_off=stim_off, date=date, monkey=monkey)
    
    if bin_function is not None:
        binned_resp = np.stack([bin_function(epoch_resp, **kwargs) for epoch_resp in true_resp])
        binned_spont = np.stack([bin_function(epoch_spont, **kwargs) for epoch_spont in true_spont])
    else:
        binned_resp = true_resp
        binned_spont = true_spont

    if shuffle is True:
        # to perform trial shuffling of checkerboard images
        indices = np.arange(len(binned_resp)) 
        # Shuffle the indices using np.random.shuffle
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        binned_resp = binned_resp[indices]
        binned_spont = binned_spont[indices]

    if raw_resp is True:
        return binned_resp.reshape(-1, resp_array.shape[1])
    else:
        norm_resp = binned_resp - np.mean(binned_spont, axis=1, keepdims=True)  
        norm_resp = norm_resp.reshape(-1, resp_array.shape[1])
        return norm_resp

def isolate_norm_spont(resp_array, date='250717', monkey='L', bin_function=None, shuffle=False, seed=None, raw_resp=False,spont_stim_off=300, **kwargs):
    """
    This Python function isolates and normalizes spontaneous responses from an array of responses, with
    options for binning, shuffling, and reshaping the data.
    
    :param resp_array: `resp_array` is a numpy array containing the response data
    :param date: The `date` parameter in the function `isolate_norm_spont` is used to specify the date
    for which the response array should be analyzed. It has a default value of '250717' if not provided
    explicitly, defaults to 250717 (optional)
    :param monkey: The `monkey` parameter in the `isolate_norm_spont` function is used to specify the
    monkey from which the response data is coming, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_norm_spont` function allows you to
    specify a function that will be applied to each epoch of spontaneous activity data before further
    processing. This function should take an epoch of spontaneous activity data as input and return the
    processed output
    :param shuffle: The `shuffle` parameter in the `isolate_norm_spont` function is a boolean flag that
    determines whether the data should be shuffled before processing. If `shuffle` is set to `True`, the
    function will shuffle the data using random indices before further processing. This can be useful
    for randomizing, defaults to False (optional)
    :param seed: The `seed` parameter in the `isolate_norm_spont` function is used to set the random
    seed for reproducibility when shuffling the indices of the binned spontaneous responses. If a
    specific `seed` value is provided, it will ensure that the same random shuffling of indices is
    :param raw_resp: The `raw_resp` parameter in the `isolate_norm_spont` function is a boolean flag
    that determines whether the function should return the binned spontaneous responses reshaped as a 2D
    array or as a 1D array, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `isolate_norm_spont` function is used
    to specify the time offset for spontaneous activity stimulation. This parameter determines the time
    at which the spontaneous activity ends. In the function, it is set to a default value of 300,
    defaults to 300 (optional)
    :return: The function `isolate_norm_spont` returns the normalized spontaneous responses based on the
    input parameters and conditions specified in the function. If `raw_resp` is True, it returns the
    reshaped binned spontaneous responses. If `shuffle` is True, it shuffles the indices before
    reshaping the binned spontaneous responses. Finally, it returns the normalized spontaneous responses
    in the shape (-1, resp
    """
    
    true_resp, true_spont = get_epoch_times(resp_array, date, spont_stim_off=spont_stim_off)

    
    if bin_function is not None:
        binned_spont = np.stack([bin_function(epoch_spont, **kwargs) for epoch_spont in true_spont])
    else:
        binned_spont = true_spont
        
    if raw_resp is True:
        return binned_spont.reshape(-1, resp_array.shape[1])
    
    if shuffle is True:
        indices = np.arange(len(binned_spont)) 
        # Shuffle the indices using np.random.shuffle
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        binned_spont = binned_spont[indices]
    norm_spont = binned_spont.reshape(-1, resp_array.shape[1])
    return norm_spont

def isolate_RS_resp(resp_array, date, open_or_closed = 'Open_eyes', monkey='L', bin_function=None, **kwargs):
    """
    This Python function isolates and processes response data based on specified parameters such as
    date, eye state, and monkey.
    
    :param resp_array: The `resp_array` parameter is likely a NumPy array containing response data. This
    array is used within the function to extract specific epochs of response data based on the provided
    indices and other parameters
    :param date: The `date` parameter in the `isolate_RS_resp` function is used to specify the date for
    which you want to isolate responses
    :param open_or_closed: The `open_or_closed` parameter in the `isolate_RS_resp` function specifies
    whether the eyes are open or closed during the response. It has a default value of 'Open_eyes', but
    you can also provide 'Closed_eyes' as an alternative value, defaults to Open_eyes (optional)
    :param monkey: The `monkey` parameter in the `isolate_RS_resp` function specifies which monkey's
    data to retrieve, defaults to L (optional)
    :param bin_function: The `bin_function` parameter in the `isolate_RS_resp` function is a function
    that can be passed as an argument to perform some operation on the epoch response data. This
    function will be applied to each epoch of the response array before it is extended to the final
    response array `resp_`
    :return: The function `isolate_RS_resp` returns the response array `resp_` after processing it based
    on the provided parameters such as the response array, date, eye state (open or closed), monkey,
    binning function, and any additional keyword arguments.
    """
    eye_query = f'state == "{open_or_closed}"'
    df_RS = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/epochs_{monkey}_RS_{date}.csv')
    df_RS_new = df_RS.query(eye_query).copy()
    df_RS_new[['t_start', 't_stop', 'dur']] = (df_RS_new[['t_start', 't_stop', 'dur']] * 1000).astype(int)

    indices = [df_RS_new['t_start'].values[e] + np.arange(df_RS_new['dur'].values[e]) for e in range(len(df_RS_new))]
    
    resp_=[]

    if bin_function is not None:
        for epoch in range(len(indices)-1):
            epoch_resp = resp_array[indices[epoch],:]
            epoch_resp = bin_function(epoch_resp, **kwargs)
            resp_.extend(epoch_resp)
            
        epoch_resp = resp_array[indices[-1][0]:,:]
        epoch_resp = bin_function(epoch_resp, **kwargs)
        resp_.extend(epoch_resp)
        
    else:
        for epoch in range(len(indices)-1):
            epoch_resp = resp_array[indices[epoch],:]
            resp_.extend(epoch_resp)
            
        epoch_resp = resp_array[indices[-1][0]:,:]
        resp_.extend(epoch_resp)

    resp_ = np.array(resp_)

    return resp_

from scipy.stats import sem

def get_img_resp_avg_sem(resp_array, date, condition_type, w_size=25,chunk_size=None,get_chunks=False):
    """
    This Python function calculates the average and standard error of the mean of neural responses based
    on input parameters such as response array, date, condition type, window size, and chunk size.
    
    :param resp_array: `resp_array` is a numpy array containing responses from neurons. The function
    `get_img_resp_avg_sem` processes this array along with other parameters to calculate the average and
    standard error of the mean (SEM) of neuron responses based on the specified conditions
    :param date: The `date` parameter is used to specify the date for which you want to analyze the
    image responses. It is likely used within the function to filter the responses based on the provided
    date
    :param condition_type: Condition type refers to the type of experimental condition or stimulus being
    presented to the neurons. It could include conditions like 'RF_spont', 'RF', 'SNR_spont', 'SNR',
    etc. These conditions may have different chunk sizes based on the type
    :param w_size: The `w_size` parameter in the `get_img_resp_avg_sem` function represents the window
    size used for calculating the chunk size based on the condition type. It is used to determine the
    number of frames in each chunk based on the condition type specified, defaults to 25 (optional)
    :param chunk_size: The `chunk_size` parameter in the `get_img_resp_avg_sem` function determines the
    size of each chunk that the response array will be split into. The size of the chunks is calculated
    based on the `condition_type` provided. If `chunk_size` is not explicitly provided, it is calculated
    :param get_chunks: The `get_chunks` parameter in the `get_img_resp_avg_sem` function is a boolean
    flag that determines whether the function should return the chunks_array or not. If `get_chunks` is
    set to `True`, the function will return the chunks_array. If `get_chunks` is set to, defaults to
    False (optional)
    :return: the average neuron response and the standard error of the mean (SEM) of the neuron
    response.
    """

    if chunk_size is None:
        if 'RF_spont' in condition_type:
            chunk_size=int(200/w_size)
        
        elif 'RF' in condition_type:
            chunk_size=int(1000/w_size)
        
        elif 'SNR_spont' in condition_type:
            chunk_size=int(200/w_size)

        elif 'SNR' in condition_type:
            chunk_size=int(400/w_size)
        else:
            get_condition_type=condition_type
    else:
        if 'spont' in condition_type:
            get_condition_type=condition_type.replace('_spont','')
        else:
            get_condition_type=condition_type
            
    n_frames = resp_array.shape[0]
    n_epochs = int(n_frames/chunk_size)

    if not is_factor(chunk_size, n_frames):
        print('Frames are not evenly split among epochs')
        del chunk_size
    chunks = np.split(resp_array[:n_epochs * chunk_size, :], n_epochs)
    chunks_array = np.array(chunks)

    if get_chunks is True:
        return chunks_array
    
    avg_neuron_resp = np.mean(chunks_array, axis=0)
    SEM_neuron_resp = sem(chunks_array, axis=0)
    return avg_neuron_resp.T, SEM_neuron_resp.T
    
def is_factor(number, n):
    return n % number == 0

def binning_with_sum(data, window_size, e=0,**kwargs):
    """
    The function `binning_with_sum` takes a dataset, divides it into windows of a specified size, and
    calculates the sum of values within each window.
    
    :param data: The `data` parameter is the input data that you want to bin. It should be a 2D numpy
    array where each row represents a data point and each column represents a feature
    :param window_size: The `window_size` parameter in the `binning_with_sum` function represents the
    size of the window over which you want to aggregate the data. It determines how many data points
    will be grouped together for summing
    :param e: The parameter `e` in the `binning_with_sum` function seems to be a placeholder variable
    that is not being used within the function. It is defined as a keyword argument with a default value
    of 0, but it is not utilized in the function logic, defaults to 0 (optional)
    :return: The function `binning_with_sum` returns binned data where each row represents the sum of
    data points within a specified window size along the columns of the input data array.
    """
    bin_datapoints = int(np.floor(len(data)/window_size))
    binned_data = np.zeros([bin_datapoints, data.shape[1]])
    for i in range(bin_datapoints):
        window_data = data[i*window_size : (i*window_size) + window_size, :]
        binned_data[i] = np.sum(window_data, axis=0)  # You can use a different aggregation function here
    return binned_data


def get_resps(condition_type='SNR', date='090817', monkey='L', w_size = 25, stim_on=0,  stim_off=400, shuffle=False, get_RF_labels=False, bin_function=binning_with_sum, keep_SNR_elecs=False, raw_resp=False, spont_stim_off=300):
    """
    This Python function retrieves neural response data based on specified conditions and electrode
    information for further analysis.
    
    :param condition_type: The `condition_type` parameter in the `get_resps` function is used to specify
    the type of condition for which you want to retrieve responses. It can take on different values
    based on the type of data you are interested in analyzing. The function uses this parameter to
    determine which data files to read, defaults to SNR (optional)
    :param date: The `date` parameter in the `get_resps` function is used to specify the date for which
    the data is being retrieved. It is a string parameter that represents the date in the format
    'MMDDYY', defaults to 090817 (optional)
    :param monkey: The `monkey` parameter in the `get_resps` function is used to specify the monkey from
    which the data is being analyzed. It is used to locate the relevant data files and metadata
    associated with the specified monkey for further processing within the function, defaults to L
    (optional)
    :param w_size: The `w_size` parameter in the `get_resps` function represents the window size for
    binning the neural responses. It is used to define the size of the time window over which the neural
    activity will be aggregated or analyzed. This parameter determines the duration of each time bin for
    processing the neural, defaults to 25 (optional)
    :param stim_on: The `stim_on` parameter in the `get_resps` function represents the time point at
    which the stimulus starts in the experiment. It is used to specify the onset time of the stimulus in
    milliseconds relative to the start of the recording, defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_resps` function represents the time point at
    which the stimulus ends during data processing. In the provided function, it is set to a default
    value of 400. This means that the stimulus ends at time point 400 in the data processing timeline,
    defaults to 400 (optional)
    :param shuffle: The `shuffle` parameter in the `get_resps` function is a boolean flag that
    determines whether the data should be shuffled before processing. If `shuffle` is set to `True`, the
    data will be randomly shuffled before further processing. If set to `False`, the data will remain in
    its, defaults to False (optional)
    :param get_RF_labels: The `get_RF_labels` parameter in the `get_resps` function is a boolean flag
    that determines whether to return condition labels along with the response arrays for the V4
    cortical area. If `get_RF_labels` is set to `True`, the function will return the response arrays for
    V4, defaults to False (optional)
    :param bin_function: The `bin_function` parameter in the `get_resps` function is used to specify the
    binning function that will be applied to the data. In this case, the default binning function being
    used is `binning_with_sum`. This function likely performs some form of binning operation on the
    :param keep_SNR_elecs: The `keep_SNR_elecs` parameter in the `get_resps` function is a boolean flag
    that determines whether to keep only the electrodes with Signal-to-Noise Ratio (SNR) less than 2,
    defaults to False (optional)
    :param raw_resp: The `raw_resp` parameter in the `get_resps` function is a boolean flag that
    determines whether the raw responses should be returned or not. If `raw_resp` is set to `True`, the
    function will return the raw responses along with any processed data. If `raw_resp` is, defaults to
    False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_resps` function is used to specify
    the time point at which the spontaneous stimulus ends. This parameter is used in the function to
    process neural response data based on the timing of the stimulus presentation and the spontaneous
    activity period. By setting `, defaults to 300 (optional)
    :return: either `resp_V4` and `resp_V1` arrays or `resp_V4`, `resp_V1`, and `cond_labels` arrays
    based on the conditions specified in the function parameters.
    """
    data_dir = main_dir + f'data/chen/monkey_{monkey}/{date}/'
    if 'RF' not in condition_type:
        SNR_df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_{date}_full.csv')
        SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_RS_{date}_removal_metadata.csv')
    else:
        SNR_df = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_SNR_250717_full.csv')
        SP = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/{monkey}_RS_250717_removal_metadata.csv')

    id_dict = pd.read_csv(main_dir + f'data/chen/metadata/monkey_{monkey}/channel_area_mapping_{monkey}.csv')
    id_dict_filtered = id_dict[id_dict['Electrode_ID'].isin(SP['Removed electrode ID'])]
    id_snr_filtered = id_dict[id_dict['Electrode_ID'].isin(SNR_df[SNR_df['SNR'] < 2]['Electrode_ID'])]


    if keep_SNR_elecs is True:
        combined_df = id_dict_filtered
    else:
        combined_df = pd.concat([id_dict_filtered, id_snr_filtered])

    SP_SNR_electrodes = {}

    for _, row in combined_df.iterrows():
        array_id = row['Array_ID']
        within_array_electrode_id = row['within_array_electrode_ID']
        SP_SNR_electrodes.setdefault(array_id, set()).add(within_array_electrode_id - 1) #subtracting for python indexing
    uncat_resp_v1 = []
    uncat_resp_v4 = []

    all_arrays = np.arange(1,17)
    NSPidc = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    data_dir = main_dir + f'data/chen/monkey_{monkey}/{date}/'
    
    if condition_type =='SNR_spont':
        array_condition_type = 'SNR'
    elif 'RS' in condition_type:
        array_condition_type='RS'
    elif 'RF' in condition_type:
        array_condition_type = 'RF'
    else:
        array_condition_type= condition_type
    for a, array in enumerate(all_arrays):
        elecs_to_delete = []
        elec_names_to_edit = np.arange(64)
        # Load the data
        with NixIO(data_dir + f'{array_condition_type}/NSP{NSPidc[a]}_array{array}_MUAe.nix', mode='ro') as io:
            block = io.read_block()

        # Finding the analog signals
        anasig = block.segments[0].analogsignals
        del block

        # Find the electrode ID annotations
        within_elec_ID = np.array(anasig[0].array_annotations['within_array_electrode_ID'])
        # Find the electrode ID annotations
        electrode_IDs = np.array(anasig[0].array_annotations['Electrode_ID'])

        array_to_edit = np.array(anasig[0])
        cortical_area = anasig[0].array_annotations['cortical_area'][0]

        del anasig

        sorted_array_to_edit = array_to_edit[:, np.argsort(within_elec_ID)]
        sorted_electrode_IDs = electrode_IDs[np.argsort(within_elec_ID)]
        elecs_to_delete = np.array([elec for elec in SP_SNR_electrodes.get(array, [])], dtype=int)

        array_to_edit = np.delete(sorted_array_to_edit, elecs_to_delete, axis=1)

        electrode_IDs = np.delete(sorted_electrode_IDs, elecs_to_delete)
        if 'V1' in cortical_area:
            clean_array = get_clean_array(array_to_edit, condition_type, date, monkey, w_size, stim_on, stim_off, bin_function=bin_function, shuffle=False, get_RF_labels=False, raw_resp=raw_resp,spont_stim_off=spont_stim_off)
            uncat_resp_v1.append(clean_array)
            del array_to_edit
        elif 'V4' in cortical_area:
            clean_array = get_clean_array(array_to_edit, condition_type, date, monkey,  w_size, stim_on, stim_off, bin_function=bin_function, shuffle=shuffle, get_RF_labels=get_RF_labels, raw_resp=raw_resp, spont_stim_off=spont_stim_off)
            if get_RF_labels is True:
                uncat_resp_v4.append(clean_array[0])
                cond_labels = clean_array[1]
            else:
                uncat_resp_v4.append(clean_array)
            del array_to_edit

    resp_V1 = np.concatenate(uncat_resp_v1, axis=1)
    resp_V4 = np.concatenate(uncat_resp_v4, axis=1)

    if get_RF_labels is True:
        return resp_V4, resp_V1, cond_labels
    return resp_V4, resp_V1


def get_clean_array(resp_array, condition_type='SNR', date='090817', monkey='L', w_size = 25, stim_on=0, stim_off=400, bin_function=binning_with_sum, shuffle=False, get_RF_labels=False, raw_resp=False, spont_stim_off=200):
    """
    This Python function takes in response data and parameters to clean and process the data based on
    different conditions such as signal-to-noise ratio (SNR), resting state (RS), and receptive field
    (RF).
    
    :param resp_array: `resp_array` is the input array containing response data. The function
    `get_clean_array` processes this array based on the specified conditions and parameters to return a
    cleaned and processed version of the data. The function handles different types of conditions such
    as SNR, RS, RF, and their variations,
    :param condition_type: The `condition_type` parameter in the `get_clean_array` function determines
    the type of data processing to be applied to the input `resp_array`. The function supports different
    condition types for processing the data, such as 'SNR', 'SNR_spont', 'RS', 'RS_open',, defaults to
    SNR (optional)
    :param date: The `date` parameter in the `get_clean_array` function is used to specify the date for
    data processing. It is a string parameter that represents the date in a specific format, such as
    '090817' in the function call, defaults to 090817 (optional)
    :param monkey: The `monkey` parameter in the `get_clean_array` function is used to specify the
    monkey from which the response data is collected. It is a string parameter that can take values like
    'L' or 'R' to indicate the left or right monkey, for example, defaults to L (optional)
    :param w_size: The `w_size` parameter in the `get_clean_array` function represents the window size
    used for binning the response data. It is an integer value that determines the size of the window
    for aggregating or summarizing the responses within each window, defaults to 25 (optional)
    :param stim_on: The `stim_on` parameter in the `get_clean_array` function represents the time point
    when the stimulus starts in the experiment. It is used in the calculation of responses based on
    different conditions such as SNR (Signal-to-Noise Ratio) or RF (Receptive Field). The value of `,
    defaults to 0 (optional)
    :param stim_off: The `stim_off` parameter in the `get_clean_array` function represents the time
    point at which the stimulus ends during data processing. It is used in various conditions like 'SNR'
    and 'RF' to determine the duration of the stimulus presentation, defaults to 400 (optional)
    :param bin_function: The `bin_function` parameter in the `get_clean_array` function is used to
    specify the function that will be applied to bin the response array data. This function will take
    the response array and window size as input parameters and return the binned data. The function can
    be customized based on the specific
    :param shuffle: The `shuffle` parameter in the `get_clean_array` function is a boolean flag that
    determines whether the data should be shuffled or not. If `shuffle` is set to `True`, the data will
    be shuffled before processing. If `shuffle` is set to `False`, the data will not, defaults to False
    (optional)
    :param get_RF_labels: The `get_RF_labels` parameter is a boolean flag that determines whether the
    function should return both the `sum_binned` array and the `binned_labels` array when set to `True`.
    If `get_RF_labels` is `True`, the function will return both arrays; otherwise, it, defaults to False
    (optional)
    :param raw_resp: The `raw_resp` parameter in the `get_clean_array` function is a boolean flag that
    specifies whether to use raw response data or not. If `raw_resp` is set to `True`, the function will
    use raw response data in the calculations. If `raw_resp` is set to `, defaults to False (optional)
    :param spont_stim_off: The `spont_stim_off` parameter in the `get_clean_array` function is used to
    specify the time point at which the spontaneous stimulus ends. This parameter is used in the
    'SNR_spont' and 'RF_spont' conditions to determine the duration of the spontaneous stimulus period,
    defaults to 200 (optional)
    :return: either `sum_binned` or a tuple containing `sum_binned` and `binned_labels` based on the
    conditions specified in the function.
    """

    if condition_type == 'SNR_spont':
        sum_binned = isolate_norm_spont(resp_array=resp_array, bin_function=bin_function, window_size=w_size, date=date, 
                                        monkey=monkey, shuffle=shuffle, raw_resp=raw_resp, spont_stim_off=spont_stim_off)
    elif condition_type == 'SNR':
        sum_binned = isolate_norm_resps(resp_array, stim_on=stim_on, 
                                        stim_off=stim_off, 
                                        bin_function=bin_function, 
                                        window_size=w_size, date=date, 
                                        monkey=monkey, shuffle=shuffle, raw_resp=raw_resp) 
    elif condition_type == 'RS':
        if bin_function is not None:
            sum_binned = bin_function(resp_array, window_size=w_size)
            
        else:
            sum_binned = resp_array
        # sum_binned -= np.mean(sum_binned,axis=0)
        
    elif condition_type == 'RS_open':
        sum_binned = isolate_RS_resp(resp_array, date, open_or_closed = 'Open_eyes', monkey=monkey, 
                                        bin_function=bin_function, window_size=w_size)
    elif condition_type == 'RS_closed':
        sum_binned = isolate_RS_resp(resp_array, date, open_or_closed = 'Closed_eyes', monkey=monkey, 
                                        bin_function=bin_function, window_size=w_size)

    elif condition_type == 'RF':
        sum_binned, binned_labels = isolate_norm_resps_RF(resp_array, stim_on=stim_on, stim_off=stim_off,
                                                        bin_function=bin_function, window_size=w_size,
                                                        date=date, monkey=monkey, raw_resp=raw_resp)
    
    elif condition_type == 'RF_spont':
        sum_binned, binned_labels = isolate_norm_spont_RF(resp_array, stim_on=stim_on, stim_off=stim_off,
                                                        bin_function=bin_function, window_size=w_size,
                                                        date=date, monkey=monkey, raw_resp=raw_resp,spont_stim_off=spont_stim_off)

    del resp_array

    if get_RF_labels is True:
        return sum_binned, binned_labels

    return sum_binned

def get_get_condition_type(condition_type):
    """
    The function `get_get_condition_type` returns a modified condition type based on specific criteria.
    
    :param condition_type: The function `get_get_condition_type` takes a `condition_type` as input and
    returns a modified version of it based on certain conditions
    :return: The function `get_get_condition_type` returns the value of the variable
    `get_condition_type`, which is determined based on the input `condition_type`. If the input contains
    both 'RF' and 'spont', it returns 'RF_spont'. If the input contains only 'RF', it returns 'RF'.
    Otherwise, it returns the original `condition_type`.
    """
    if 'RF' in condition_type and 'spont' in condition_type:
        get_condition_type='RF_spont'
    elif 'RF' in condition_type:
        get_condition_type='RF'
    else:
        get_condition_type=condition_type
    return get_condition_type


