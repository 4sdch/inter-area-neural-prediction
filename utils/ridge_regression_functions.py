
import numpy as np
from scipy import stats
import copy


main_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/inter_areal_predictability/'
func_dir = '/Users/diannahidalgo/Documents/thesis_shenanigans/aim2_project/inter_areal_predictability/utils/'

import sys
sys.path.insert(0,func_dir)


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from scipy import stats as st

# will's functions
def pearsonr2(a, b):
    for x in (a, b):
        if x.std()==0:
            return np.nan
        mu = x.mean()
        if np.linalg.norm(x - mu) < 1e-13 * np.abs(mu):
            return np.nan
    return st.pearsonr(a, b)[0]

def worker_function(X, y, alpha, train_idc, test_idc, frames_reduced=None, control_shuffle=False):
    if frames_reduced is not None:
        train_idc_reduced_x = copy.deepcopy(train_idc[frames_reduced:-frames_reduced])
        train_idc_reduced_y = train_idc[frames_reduced:-frames_reduced]
        test_idc_x = copy.deepcopy(test_idc)
        test_idc_y = copy.deepcopy(test_idc)
    else:
        train_idc_reduced_x = train_idc.copy()
        train_idc_reduced_y = train_idc.copy()  
    
    if control_shuffle is True:
        np.random.shuffle(train_idc_reduced_x)
        np.random.shuffle(test_idc_x)
   
    Xtrain = X[train_idc_reduced_x]
    ytrain = y[train_idc_reduced_y]
    Xval = X[test_idc_x]
    yval=y[test_idc_y]

    model4fit = Ridge(alpha=alpha, max_iter=10000)
    model4fit.fit(Xtrain, ytrain)

    ypred=model4fit.predict(Xval)

    corr = np.array([pearsonr2(*vs) for vs in zip(yval.T, ypred.T)])
    split_evars = np.square(corr / 1)    
    return ypred, test_idc, model4fit.coef_,split_evars

def get_predictions_evars_parallel(layer_used, layer_to_predict, alpha, n_splits=10, frames_reduced=None, 
                                   verbose=None, save_weights=False, standarize_X=True, mean_subtract_y=False, 
                                   control_shuffle=False, mean_split_evars=True):
    y = layer_to_predict
    if standarize_X is True:
        scaler = StandardScaler()
        X = scaler.fit_transform(layer_used)
    else:
        X=layer_used     
    if verbose == 1:
        print('dataset shape', X.shape, y.shape)

    kf = KFold(n_splits=n_splits, shuffle=False)

    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    y_preds, f_indices, split_evars = [], [],[]
    all_coefs = np.zeros([n_splits, y.shape[1], X.shape[1]])

    results = Parallel(n_jobs=-1)(delayed(worker_function)(X, y, alpha, train_idc, test_idc, frames_reduced, control_shuffle=control_shuffle) for train_idc, test_idc in kf.split(X))
    for t, (ypred, test_idc, coef, split_evar) in enumerate(results):
        y_preds.append(ypred)
        f_indices.append(test_idc)
        split_evars.append(split_evar)
        if save_weights:
            all_coefs[t] = coef

    y_preds = np.concatenate(y_preds, axis=0)
    f_indices = np.concatenate(f_indices, axis=None)
    sorted_preds= y_preds[np.argsort(f_indices)]
    corr = np.array([pearsonr2(*vs) for vs in zip(y.T, sorted_preds.T)])
    evars = np.square(corr / 1)
    
    if verbose == 1:
        print(f'mean_alpha: {np.nanmean(evars)}')

    if save_weights:
        return sorted_preds, evars, all_coefs, split_evars
    elif mean_split_evars is True:
        return sorted_preds, np.nanmean(np.array(split_evars), axis=0)
    else:
        return sorted_preds, evars

def get_best_alpha_evars(layer_to_use, layer_to_predict, n_splits=10, frames_reduced = 5, 
                         alphas=None, silence=None, standardize_X=True, control_shuffle=False,mean_split_evars=True):

    alpha_evars = []
    if alphas is None:
        alphas = [5e2,1e3,5e3,1e4,5e4,1e5,5e5,1e6,5e6,1e7]

    if len(layer_to_predict.shape)>1:
        n_neurons = layer_to_predict.shape[1]
    else:
        n_neurons = 1
        
    all_alpha_evars = np.zeros([len(alphas),n_neurons])

    for count, alpha in enumerate(alphas):
#         print(f'{alpha_list}')

        _, evars = get_predictions_evars_parallel(layer_to_use,layer_to_predict, n_splits=n_splits,
                                             alpha=alpha,verbose=0, frames_reduced=frames_reduced, standarize_X=standardize_X,
                                             control_shuffle=control_shuffle,mean_split_evars=mean_split_evars)

        all_alpha_evars[count] = np.array(evars)
        if silence is not None:
            print(f'alpha {alphas[count]}, EV: {np.nanmean(all_alpha_evars[count])}')
        alpha_evars.append(all_alpha_evars[count])
        if len(alpha_evars)>3:
            if np.nanmean(alpha_evars[-1])< np.nanmean(alpha_evars[-4]):
                break
    best_mean_alpha= np.argmax(np.nanmean(all_alpha_evars, axis=1))
    chosen_alpha_list=alphas[best_mean_alpha]
    if silence is not None:
        print(f'Best alpha: {chosen_alpha_list}, EV: {np.nanmean(all_alpha_evars[best_mean_alpha])}')
    return chosen_alpha_list, all_alpha_evars[best_mean_alpha]

