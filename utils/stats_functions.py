import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def perm_test(group1, group2):
    """Permutation test for independent samples.

    This function computes the p-value for a two-sample permutation test
    based on the difference in means between two independent groups.

    Args:
        group1 (array-like): Data for group 1.
        group2 (array-like): Data for group 2.

    Returns:
        float: p-value for the permutation test.
    """
    # Observed test statistic (e.g., difference in means)
    observed_statistic = np.nanmean(group1) - np.nanmean(group2)
    # Number of permutations to perform
    num_permutations = 10000
    # Create an array to store the permuted test statistics
    permuted_statistics = np.zeros(num_permutations)
    # Combine the data from both groups
    combined_data = np.concatenate((group1, group2))
    # Perform the permutation test
    for i in range(num_permutations):
        # Randomly shuffle the combined data
        np.random.shuffle(combined_data)
    
        # Split the shuffled data back into two groups
        permuted_group1 = combined_data[:len(group1)]
        permuted_group2 = combined_data[len(group1):]
        
        # Calculate the test statistic for this permutation
        permuted_statistic = np.nanmean(permuted_group1) - np.nanmean(permuted_group2)
        
        # Store the permuted test statistic
        permuted_statistics[i] = permuted_statistic

    # Calculate the p-value by comparing the observed statistic to the permuted distribution
    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()

    return p_value

def perm_test_paired(group1, group2):
    """Permutation test for paired samples.

    This function computes the p-value for a paired permutation test
    based on the difference in means between two paired groups.

    Args:
        group1 (array-like): Data for group 1.
        group2 (array-like): Data for group 2.

    Returns:
        float: p-value for the paired permutation test.
    """
    # Observed test statistic (e.g., difference in means)
    observed_statistic = np.nanmean(group2-group1)

    # Number of permutations to perform
    num_permutations = 10000

    # Create an array to store the permuted test statistics
    permuted_statistics = np.zeros(num_permutations)

    # Combine the differences
    pooled_differences = group2-group1
    
    # Perform the permutation test
    for i in range(num_permutations):
        # shuffle differences
        permuted_differences = pooled_differences * np.random.choice([-1, 1], size=len(pooled_differences))
        
        # Recalculate mean difference for the permuted dataset
        permuted_mean_difference = np.nanmean(permuted_differences)
    
        # Store the permuted mean difference
        permuted_statistics[i] = permuted_mean_difference

    # Calculate the p-value by comparing the observed statistic to the permuted distribution
    p_value = (np.abs(permuted_statistics) >= np.abs(observed_statistic)).mean()

    return p_value

# Function to perform hierarchical permutation test with animal bootstrapping
def hierarchical_permutation_test(data, mouse_or_date, dependent_variable, neuron_property,perm_type='ind', num_permutations=1000):
    observed_statistic = calculate_statistic(data, dependent_variable, neuron_property, perm_type)  # Replace with your actual calculation
    """Hierarchical permutation test with animal bootstrapping.

    This function performs a hierarchical permutation test with animal bootstrapping.
    It calculates a statistic of interest for the observed data and then generates
    permuted datasets by bootstrapping animals. The p-value is computed by comparing
    the observed statistic to the distribution of permuted statistics.

    Args:
        data (pandas.DataFrame): Input DataFrame containing the data.
        mouse_or_date (str): Identifier for mouse or date.
        dependent_variable (str): Dependent variable.
        neuron_property (str): Neuron property.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        num_permutations (int, optional): Number of permutations. Defaults to 1000.

    Returns:
        float: p-value for the hierarchical permutation test.
    """
    # Create an empty array to store permuted statistics
    permuted_statistics = np.zeros(num_permutations)

    # Iterate through each permutation
    for i in range(num_permutations):
        # Bootstrap animals (resample entire animals with replacement)
        bootstrap_animals_or_dates = np.random.choice(data[mouse_or_date].unique(), size=len(data[mouse_or_date].unique()), replace=True)
        data2 = data[data[mouse_or_date].isin(bootstrap_animals_or_dates)]
        if 'Mouse' in mouse_or_date:
        # bootstrapped_data = data[data[mouse_or_date].isin(bootstrap_animals_or_dates)]
            min_cells_per_mouse = min(data[data[mouse_or_date].isin(bootstrap_animals_or_dates)].groupby(['Mouse',dependent_variable])[neuron_property].count())
            bootstrapped_data = pd.concat([group_.sample(min_cells_per_mouse, replace=False) for _, group_ in data2.groupby(['Mouse',dependent_variable])])
        else:
            min_cells_per_date = min(data[data[mouse_or_date].isin(bootstrap_animals_or_dates)].groupby([mouse_or_date,dependent_variable])[neuron_property].count())
            bootstrapped_data = pd.concat([group_.sample(min_cells_per_date, replace=False) for _, group_ in data2.groupby([mouse_or_date,dependent_variable])])

        if perm_type =='ind':
            # Permute values within each bootstrapped animal
            for animal in bootstrapped_data[mouse_or_date].unique():
                animal_values = bootstrapped_data.loc[bootstrapped_data[mouse_or_date] == animal, neuron_property].values
                np.random.shuffle(animal_values)
                bootstrapped_data.loc[bootstrapped_data[mouse_or_date] == animal, neuron_property] = animal_values
            # Calculate the permuted statistic
            permuted_statistic = calculate_statistic(bootstrapped_data, dependent_variable, neuron_property, perm_type=perm_type)
        elif perm_type =='paired':
            permuted_statistic = calculate_statistic(bootstrapped_data, dependent_variable, neuron_property, perm_type=perm_type, paired_shuffle=True)
        # Store the permuted statistic
        permuted_statistics[i] = permuted_statistic
    # Calculate the p-value
    p_value = np.mean(np.abs(permuted_statistics) >= np.abs(observed_statistic))
    return p_value


def get_one_way_anova_pstats(df,variable1,variable1_order, neuron_property,perm_t=False, perm_type='ind'):
    """
    Perform one-way ANOVA and pairwise post-hoc tests between groups based on a given variable.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    variable1 : str
        The name of the column representing the independent variable.
    variable1_order : list
        The order of groups for the independent variable.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    perm_t : bool, optional
        Whether to perform permutation tests instead of traditional t-tests (default is False).
    perm_type : {'ind', 'paired'}, optional
        Type of permutation test to perform. 'ind' for independent samples permutation test, 'paired' for paired samples permutation test (default is 'ind').
    Returns:
    --------
    p_val_names : list
        List of names for pairwise comparisons.
    adjusted_p_values : list
        List of adjusted p-values for pairwise comparisons after applying the Benjamini-Hochberg correction.

    Notes:
    ------
    This function performs one-way ANOVA and subsequent pairwise post-hoc tests between groups based on the given independent variable.
    It calculates either traditional t-tests or permutation tests depending on the specified parameters.
    """
    df_posthoc = df.dropna().copy()
    # Perform pairwise t-tests with Benjamini-Hochberg correction
    groups = variable1_order
    p_values = []
    p_val_names = []

    for group1, group2 in combinations(groups, 2):
        group1_data = df_posthoc[df_posthoc[f'{variable1}'] == group1][neuron_property]
        group2_data = df_posthoc[df_posthoc[f'{variable1}'] == group2][neuron_property]
        
        if perm_type == 'paired':
            p_value = perm_test_paired(group1_data, group2_data)
        elif perm_t is True:
            p_value = perm_test(group1_data, group2_data)
        else:
            _, p_value = stats.ttest_ind(group1_data, group2_data)
        p_values.append(p_value)
        p_val_names.append(group1 + '_' + group2)

    # Apply Benjamini-Hochberg correction
    adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

    return p_val_names, adjusted_p_values

def get_oneway_anova_stars(df_, dependent_variable,dependent_variable_order, neuron_property, perm_t=True, perm_type='ind'):
    """
    Perform one-way ANOVA and generate significance stars for pairwise comparisons.

    Parameters:
    -----------
    df_ : pandas.DataFrame
        The DataFrame containing the data.
    dependent_variable : str
        The name of the column representing the independent variable.
    dependent_variable_order : list
        The order of groups for the independent variable.
    neuron_property : str
        The name of the column representing the dependent variable (property of neurons).
    perm_t : bool, optional
        Whether to perform permutation tests instead of traditional t-tests (default is True).
    perm_type : {'ind', 'paired'}, optional
        Type of permutation test to perform. 'ind' for independent samples permutation test, 'paired' for paired samples permutation test (default is 'ind').
    
    Returns:
    --------
    p_val_names : list
        List of names for pairwise comparisons.
    all_stars : list
        List of significance stars indicating the level of significance for each pairwise comparison.

    Notes:
    ------
    This function performs one-way ANOVA and generates significance stars based on the adjusted p-values for pairwise comparisons.
    The level of significance is indicated by the number of stars: *** for p < 0.001, ** for p < 0.01, * for p < 0.05, and 'n.s.' for not significant.
    """
    
    all_stars = []
    p_val_names, adjusted_p_values= get_one_way_anova_pstats(df_,dependent_variable,dependent_variable_order, 
                                                            neuron_property,perm_t=perm_t, perm_type=perm_type)
    for name, p_value in zip(p_val_names, adjusted_p_values):
        if p_value <1e-3:
            stars = '***'
        elif p_value <1e-2:
            stars = '**'
        elif p_value <0.05:
            stars='*'
        else:
            stars='n.s.'
        all_stars.append(stars)
    return p_val_names, all_stars

# Example function for the statistic of interest
def calculate_statistic(data, group, neuron_property, perm_type='ind', paired_shuffle=False):
    """Calculate the statistic of interest.

    This function calculates the statistic of interest based on the input data.

    Args:
        data (pandas.DataFrame): Input DataFrame containing the data.
        group (str): Group identifier.
        neuron_property (str): Neuron property.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        paired_shuffle (bool, optional): Whether to perform paired shuffling. Defaults to False.

    Returns:
        float: Calculated statistic.
    """
    groups = data[group].unique()
    if perm_type =='ind':
        mean_group_a = data[data[group] == groups[0]][neuron_property].mean()
        mean_group_b = data[data[group] == groups[1]][neuron_property].mean()
        return mean_group_a - mean_group_b
    elif perm_type =='paired':
        if data[data[group] == groups[0]][neuron_property].size != data[data[group] == groups[1]][neuron_property].size:
            print('sizes are not the same, you should not used a paired permutation test here')
            print(data[data[group] == groups[0]][neuron_property].size,data[data[group] == groups[1]][neuron_property].size)
        pooled_differences = data[data[group] == groups[0]][neuron_property].values-data[data[group] == groups[1]][neuron_property].values
        if paired_shuffle is True:
            permuted_differences = pooled_differences * np.random.choice([-1, 1], size=len(pooled_differences))
            # Recalculate mean difference for the permuted dataset
            return np.nanmean(permuted_differences)
        else:
            return np.nanmean(pooled_differences)
        
def get_t_test_stars(df_, dependent_variable, neuron_property, print_pval=False, 
                    perm_t=True, perm_type='ind', hierarchical=False, num_permutations=1000, mouse_or_date='Mouse_Name'):
    """Perform t-test and return significance stars.

    This function conducts a t-test between two groups defined by a dependent variable,
    computes the p-value, and assigns significance stars based on the p-value.

    Args:
        df_ (pandas.DataFrame): Input DataFrame containing the data.
        dependent_variable (str): Dependent variable defining the groups.
        neuron_property (str): Neuron property to compare between groups.
        print_pval (bool, optional): Whether to print the p-value. Defaults to False.
        perm_t (bool, optional): Whether to perform a permutation test. Defaults to True.
        perm_type (str, optional): Type of permutation test ('ind' for independent or 'paired' for paired). Defaults to 'ind'.
        hierarchical (bool, optional): Whether to perform hierarchical permutation test. Defaults to False.
        num_permutations (int, optional): Number of permutations. Defaults to 1000.

    Returns:
        str: Significance stars indicating the level of significance.
    """
    variables = df_[dependent_variable].unique()
    group_1 =df_[df_[dependent_variable]==variables[0]][neuron_property].dropna().values
    group_2 =df_[df_[dependent_variable]==variables[1]][neuron_property].dropna().values
    
    if hierarchical is True:
        p_value = hierarchical_permutation_test(df_,mouse_or_date=mouse_or_date, 
                                        dependent_variable=dependent_variable, 
                                        neuron_property=neuron_property,
                                        perm_type=perm_type,num_permutations=num_permutations)
        
    elif perm_type=='paired':
        p_value = perm_test_paired(group_1, group_2)
    elif perm_t is True:
        p_value = perm_test(group_1, group_2)
    elif perm_type=='ind':
        _, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
    else:
        print('perm_type must be either ind or paired')
        return np.nan
    if p_value <1e-3:
        stars = '***'
    elif p_value <1e-2:
        stars = '**'
    elif p_value <0.05:
        stars='*'
    else:
        stars='n.s.'
    if print_pval is True:
        print(p_value)
    return stars

