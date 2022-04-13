# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:21:44 2021

@author: Fabian
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def subdivide_dataframe_by_feature(initial_dataframe,
                                   feature="unit_number",
                                   drop_feature=False):

    list_of_dataframes = []
    for _, data in initial_dataframe.groupby(feature):
        remaining_features = data.columns
        if drop_feature:
            remaining_features = remaining_features.drop([feature])
        list_of_dataframes.append(data[remaining_features])

    return list_of_dataframes


def add_target_feature(dataframe,
                       regression_feature="time_in_cycles",
                       target_feature_name="target",
                       unit=pd.Timedelta("1 h"),
                       target_clip_lower=False,
                       target_min=10,
                       target_clip_upper=False,
                       target_max=125):

    target_series = dataframe[regression_feature]
    target_series = target_series.iloc[-1] - target_series
    if type(target_series.iloc[0]) == pd.Timedelta:
        target_series /= unit

    if target_clip_lower:
        target_series = target_series.clip(lower=target_min)
    if target_clip_upper:
        target_series = target_series.clip(upper=target_max)

    dataframe[target_feature_name] = target_series.astype("float32")

    '''
    # For further preprocessing in future: set 'target_min' and 'target_max'
    # dependant from 'regression_feature'

    reverse_target_series = pd.Series(target_series.iloc[::-1],
                                      index=target_series.index).astype("float32")
    '''

    return dataframe


def create_coherent_windows(dataframe,
                            target_feature="target",
                            max_gap=1,
                            min_window=100,
                            print_output=False):

    target_series = dataframe[target_feature]
    total_len = len(target_series)

    # Determine indices for separation
    delta_series = target_series.values[:-1] - target_series.iloc[1:]
    # For taking values by iloc-index it's necessary to be sure that indices are ascending
    delta_series.index = list(range(1, total_len))

    separation_indices = list(delta_series[delta_series > max_gap].index)
    #print("Separation iloc indices (before):\n", separation_indices)

    # Case 1: dataframe is already coherent; only one window which is identical to the dataframe
    if len(separation_indices) == 0:
        if print_output:
            print(
                "There's no gap between single measurements and splitting in windows is not necessary\n")
        if total_len < min_window:
            print("DataFrame can't be used")
            return None
        return [dataframe]

    # Case 2: dataframe is not coherent and dividing the dataframe in multiple coherent windows is necessary
    if separation_indices[0] != 0:
        separation_indices.insert(0, 0)
    if separation_indices[-1] != total_len - 1:
        separation_indices.append(total_len)
    #print("Separation iloc indices (after):\n", separation_indices)

    window_list = []
    for index, end_index in enumerate(separation_indices[1:]):
        start_index = separation_indices[index]
        window = dataframe.iloc[start_index:end_index]
        window_len = len(window)

        if print_output:
            print("\nStart (iloc):\t", start_index,
                  "\tEnd (iloc) - exclusive:\t", end_index, "\tWidth:\t", window_len)
            print("Start (loc):\t",
                  window.index[0], "\tEnd (loc) - inclusive:\t", window.index[-1])

        # Do not append to list if size of window is too small
        if window_len < min_window:
            print("Window can't be used due to window lenght")
            continue

        window_list.append(window)

    print("\nMax. possible windows:\t", len(separation_indices) - 1)
    print("Created windows:\t", len(window_list))

    return window_list


def get_subwindows_from_window(window_df,
                               features,
                               target_feature="target",
                               subwindows=32,
                               subwindow_length=50,
                               subwindow_target="max",
                               maximize_subwindows=False,
                               shuffle=True,
                               seed_value=None):

    if type(shuffle) is not bool:
        raise ValueError("Only 'True' and 'False' are valid values for argument 'shuffle'")

    if type(maximize_subwindows) is not bool:
        raise ValueError("Only 'True' and 'False' are valid values for argument 'maximize_subwindows'")

    len_window = len(window_df)
    if len_window < subwindow_length:
        print("Window has not enough samples and can't be used")
        return None, None

    # Define number of subwindows created out of 'window_df'
    if maximize_subwindows:
        number_of_subwindows = len_window - subwindow_length
    else:
        number_of_subwindows = subwindows
        np.random.seed(seed_value)
        start_indices = sorted(np.random.choice(np.arange(0, len_window - subwindow_length),
                                                size=subwindows,
                                                replace=False))
        # print(start_indices)

    feature_values_per_window = []
    target_values_per_window = []
    for i in range(number_of_subwindows):

        # Create subwindows starting from first or from a random position
        if maximize_subwindows:
            start_index = i
        else:
            start_index = start_indices[i]

        end_index = start_index + subwindow_length
        subwindow = window_df.iloc[start_index:end_index]
        assert len(subwindow) == subwindow_length

        # Take feature(s) from subwindows
        feature_values = np.array(subwindow[features].values)
        feature_values_per_window.append(feature_values)

        # Define target values
        target_series = subwindow[target_feature]
        if subwindow_target == "min":
            target_value = target_series.min()
        elif subwindow_target == "max":
            target_value = target_series.max()
        elif subwindow_target == "mean":
            target_value = target_series.mean()
        else:
            raise ValueError(
                "%s is no valid value for 'subwindow_target'" % subwindow_target)
        target_values_per_window.append(target_value)

    # Transpose all single ndarray: change position of feature values with position of samples
    #print("Shape before adjustment:\t", np.array(feature_values_per_window).shape)
    feature_values_per_window = np.array(feature_values_per_window).transpose((0, 2, 1)).astype(np.float32)
    #print("Shape after adjustment:\t", np.array(feature_values_per_window).shape)

    # Make each entry of list to a sublist
    target_values_per_window = np.array(target_values_per_window)[..., np.newaxis].astype(np.float32)

    # Shuffle 'feature_values_per_window' and 'target_values_per_window'
    if shuffle:
        shuffle_seed = 42 # seed_value if not None else np.random.randint(1000)
        np.random.seed(shuffle_seed)
        np.random.shuffle(feature_values_per_window)
        np.random.seed(shuffle_seed)
        np.random.shuffle(target_values_per_window)
        np.random.seed(None)  # otherwise following seed would always be the same

    return feature_values_per_window, target_values_per_window


def get_subwindows_from_dataframes(list_of_dataframes,
                                   features,
                                   target_feature="target",
                                   window_length=100,
                                   subwindows=32,
                                   subwindow_length=50,
                                   subwindow_target="max",
                                   maximize_subwindows=False,
                                   shuffle=True):

    kwargs = dict(features=features,
                  target_feature=target_feature,
                  subwindows=subwindows,
                  subwindow_length=subwindow_length,
                  subwindow_target=subwindow_target,
                  maximize_subwindows=maximize_subwindows,
                  shuffle=shuffle)

    all_feature_values = []
    all_target_values = []
    for dataframe in tqdm(list_of_dataframes, desc="Generating Subwindows"):
        window_list = create_coherent_windows(dataframe,
                                              min_window=window_length)
        if window_list is None:
            continue
        for window in window_list:
            feature_values_per_window, target_values_per_window = get_subwindows_from_window(window, **kwargs)
            if feature_values_per_window is not None and target_values_per_window is not None:
                all_feature_values.append(feature_values_per_window)
                all_target_values.append(target_values_per_window)

    # np.concatentate also possible
    all_feature_values = np.vstack(all_feature_values)
    # np.concatentate also possible
    all_target_values = np.vstack(all_target_values)

    return all_feature_values, all_target_values


def subdivide_indices_of_a_list(all_list_indices,
                                first_list_proportion=0.7,
                                fixed_seed=True,
                                sort_lists=False,
                                verbose=False):

    first_list_size = int(first_list_proportion * len(all_list_indices))

    # Take random values from 'all_indices';
    # duplicates are not possible, since using argument in np.random.choice: replace=False
    # Finally, 'all_indices' will be separated into two lists

    if fixed_seed:
        np.random.seed(42)
    first_list_indices = np.random.choice(all_list_indices,
                                          size=first_list_size,
                                          replace=False)

    # Take only values from units which are not part of test_samples
    # (= values from units minus values from test_samples)

    second_list_indices = np.setdiff1d(all_list_indices,
                                       first_list_indices)
    if verbose:
        print("Length 'all_indices':\t\t\t", len(all_list_indices))
        print("Length 'first_list_indices':\t\t", len(first_list_indices))
        print("Length 'second_list_indices':\t\t", len(second_list_indices))
        print("Proportion 'first_list_indices' [%]:\t", round(
            100 * len(first_list_indices) / len(all_list_indices), 2))
        print("Proportion 'second_list_indices' [%]:\t", round(
            100 * len(second_list_indices) / len(all_list_indices), 2))

    if sort_lists:
        first_list_indices.sort()
        second_list_indices.sort()

    return first_list_indices, second_list_indices


def normalize_feature_values(features_array,
                             train_standardization_params=None):

    # Important to not overwrite 'features_array'
    features_array_norm = np.array(features_array)

    invalid_features = []
    standardization_params = []
    for feature_index in range(features_array_norm.shape[1]):
        feature_data = features_array_norm[:, feature_index, :]

        # Check for (default argument and all feature data identical) or (not default and list entry is None)
        if (train_standardization_params is None and (feature_data[0][0] == feature_data).all()) or\
                (train_standardization_params is not None and train_standardization_params[feature_index] is None):

            invalid_features.append(feature_index)
            standardization_params.append(None)
            continue

        if train_standardization_params is not None and train_standardization_params[feature_index] is not None:
            mean = train_standardization_params[feature_index][0]
            std = train_standardization_params[feature_index][1]

        # Normalize feature data
        else:
            mean = feature_data.mean()
            std = feature_data.std()

        standardization_params.append([mean, std])
        feature_data = (feature_data - mean) / std

        features_array_norm[:, feature_index, :] = feature_data

    # If all features have same values then it's necessary to raise an error, because
    # it'd be necessary to delete from all_targets too
    assert len(invalid_features) != features_array_norm.shape[1]

    features_array_norm = np.delete(features_array_norm,
                                    invalid_features,
                                    axis=1)
    if len(invalid_features) > 0:
        print("Len invalid features:\t", len(invalid_features))
        print("Invalid features are:\t", invalid_features, "\n")

    return features_array_norm, standardization_params


def pipeline_level_0(input_file,
                     subwindow_features,
                     min_window_length=100,
                     subwindows=32,
                     subwindow_length=50,
                     subwindow_target="max",
                     maximize_subwindows=False,
                     shuffle_subwindows=True,
                     percentage_training=70,
                     percentage_test=15,
                     splitting_feature="unit_number",
                     regression_feature="time_in_cycles",
                     target_feature_name="target",
                     target_unit=pd.Timedelta("1 h"),
                     target_clip_lower=False,
                     target_min=10,
                     target_clip_upper=False,
                     target_max=125,
                     fixed_dataframe_assignment=True,
                     sort_dataframes_after_splitting=False,
                     verbose=True):
    '''

    Parameters
    ----------
    input_file : TYPE
        DESCRIPTION.
    subwindow_features : TYPE
        DESCRIPTION.
    min_window_length : TYPE, optional
        DESCRIPTION. The default is 100.
    subwindows : TYPE, optional
        DESCRIPTION. The default is 30.
    subwindow_length : TYPE, optional
        DESCRIPTION. The default is 50.
    subwindow_target : TYPE, optional
        DESCRIPTION. The default is "max".
        Take 'min', 'max' or 'mean' of a subwindow's target values
    maximize_subwindows : TYPE, optional
        DESCRIPTION. The default is True.
    shuffle_subwindows : TYPE, optional
        DESCRIPTION. The default is True.
    percentage_training : TYPE, optional
        DESCRIPTION. The default is 70.
    percentage_test : TYPE, optional
        DESCRIPTION. The default is 15.
    splitting_feature : TYPE, optional
        DESCRIPTION. The default is "unit_number".
    regression_feature : TYPE, optional
        DESCRIPTION. The default is "time_in_cycles".
    target_feature_name : TYPE, optional
        DESCRIPTION. The default is "target".
    target_unit : TYPE, optional
        DESCRIPTION. The default is pd.Timedelta("1 h").
    target_clip_lower : TYPE, optional
        DESCRIPTION. The default is False.
    target_min : TYPE, optional
        DESCRIPTION. The default is 10.
    target_clip_upper : TYPE, optional
        DESCRIPTION. The default is False.
    target_max : TYPE, optional
        DESCRIPTION. The default is 125.
    fixed_dataframe_assignment : TYPE, optional
        DESCRIPTION. The default is True.
        Assignment of dataframes to training, test and validation is random or fixed
    sort_dataframes_after_splitting : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    X_train_norm : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    time_in_cycles_train : TYPE
        DESCRIPTION.
    unit_train : TYPE
        DESCRIPTION.
    X_test_norm : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.
    time_in_cycles_test : TYPE
        DESCRIPTION.
    unit_test : TYPE
        DESCRIPTION.
    X_val_norm : TYPE
        DESCRIPTION.
    y_val : TYPE
        DESCRIPTION.
    time_in_cycles_val : TYPE
        DESCRIPTION.
    unit_val : TYPE
        DESCRIPTION.
    standardization_params : TYPE
        DESCRIPTION.

    '''

    if percentage_test + percentage_test > 100:
        raise ValueError("More than 100 % of data can't be splitted")

    print("Chosen splitting [%]:")
    print("Training:          ", percentage_training)
    print("Test:              ", percentage_test)
    print("Validation:        ", 100 - (percentage_training + percentage_test))

    # Create list of dataframes
    list_of_dataframes = subdivide_dataframe_by_feature(input_file,
                                                        feature=splitting_feature)

    # Add target values to each dataframe
    for dataframe in list_of_dataframes:
        add_target_feature(dataframe,
                           regression_feature=regression_feature,
                           target_feature_name=target_feature_name,
                           unit=target_unit,
                           target_clip_lower=target_clip_lower,
                           target_min=target_min,
                           target_clip_upper=target_clip_upper,
                           target_max=target_max)

    # Create lists to chose dataframes with corresponding index for:
    # - training   (-> first_list_indices)
    # - test       (-> second_list_indices)
    # - validation (-> third_list_indices)

    list_indices = list(range(len(list_of_dataframes)))

    proportion_list_1 = percentage_training * 1e-2
    proportion_list_2 = percentage_test / (100 - percentage_training)

    first_list_indices,\
        second_list_indices = subdivide_indices_of_a_list(list_indices,
                                                          first_list_proportion=proportion_list_1,
                                                          fixed_seed=fixed_dataframe_assignment,
                                                          sort_lists=sort_dataframes_after_splitting)

    #print("'first_list_indices':                     ", first_list_indices)
    #print("'second_list_indices' (before splitting): ", second_list_indices)

    second_list_indices,\
        third_list_indices = subdivide_indices_of_a_list(second_list_indices,
                                                         first_list_proportion=proportion_list_2,
                                                         fixed_seed=fixed_dataframe_assignment,
                                                         sort_lists=sort_dataframes_after_splitting)

    #print("'second_list_indices' (after splitting):  ", second_list_indices)
    #print("'third_list_indices':                     ", third_list_indices)

    subwindowing_kwargs = dict(window_length=min_window_length,
                               subwindows=subwindows,
                               subwindow_length=subwindow_length,
                               subwindow_target=subwindow_target,
                               maximize_subwindows=maximize_subwindows,
                               shuffle=shuffle_subwindows)

    # Training dataset
    time_in_cycles_train,\
        y_train = get_subwindows_from_dataframes([list_of_dataframes[i] for i in first_list_indices],
                                                 features=[regression_feature],
                                                 target_feature=target_feature_name,
                                                 **subwindowing_kwargs)

    time_in_cycles_train = np.concatenate(time_in_cycles_train).astype(int)

    X_train,\
        unit_train = get_subwindows_from_dataframes([list_of_dataframes[i] for i in first_list_indices],
                                                    features=subwindow_features,
                                                    target_feature=splitting_feature,
                                                    **subwindowing_kwargs)

    unit_train = unit_train.astype(int)

    # Test dataset
    time_in_cycles_test,\
        y_test = get_subwindows_from_dataframes([list_of_dataframes[i] for i in second_list_indices],
                                                features=[regression_feature],
                                                target_feature=target_feature_name,
                                                **subwindowing_kwargs)

    time_in_cycles_test = np.concatenate(time_in_cycles_test).astype(int)

    X_test,\
        unit_test = get_subwindows_from_dataframes([list_of_dataframes[i] for i in second_list_indices],
                                                   features=subwindow_features,
                                                   target_feature=splitting_feature,
                                                   **subwindowing_kwargs)

    unit_test = unit_test.astype(int)

    # Validation dataset
    time_in_cycles_val,\
        y_val = get_subwindows_from_dataframes([list_of_dataframes[i] for i in third_list_indices],
                                               features=[regression_feature],
                                               target_feature=target_feature_name,
                                               **subwindowing_kwargs)

    time_in_cycles_val = np.concatenate(time_in_cycles_val).astype(int)

    X_val,\
        unit_val = get_subwindows_from_dataframes([list_of_dataframes[i] for i in third_list_indices],
                                                  features=subwindow_features,
                                                  target_feature=splitting_feature,
                                                  **subwindowing_kwargs)

    unit_val = unit_val.astype(int)

    # Normalization
    X_train_norm,\
        standardization_params = normalize_feature_values(
            features_array=X_train)

    X_test_norm, _ = normalize_feature_values(features_array=X_test,
                                              train_standardization_params=standardization_params)

    X_val_norm, _ = normalize_feature_values(features_array=X_val,
                                             train_standardization_params=standardization_params)

    print("Number of dataframes:                    ",
          len(list_of_dataframes), end="\n\n")
    #print("Shape 'X_train' before normalization:    ", X_train.shape)
    print("Shape 'X_train' after normalization:     ", X_train_norm.shape)
    #print("Shape 'y_train':                         ", y_train.shape)
    #print("Shape 'time_in_cycles_train':            ", time_in_cycles_train.shape)
    #print("Shape 'unit_train':                      ", unit_train.shape, end="\n\n")

    #print("Shape 'X_test' before normalization:     ", X_test.shape)
    print("Shape 'X_test_norm' after normalization: ", X_test_norm.shape)
    #print("Shape 'y_test':                          ", y_test.shape)
    #print("Shape 'time_in_cycles_test':             ", time_in_cycles_test.shape)
    #print("Shape 'unit_test':                       ", unit_test.shape, end="\n\n")

    #print("Shape 'X_val' before normalization:      ", X_val.shape)
    print("Shape 'X_val_norm' after normalization:  ", X_val_norm.shape)
    #print("Shape 'y_val':                           ", y_val.shape)
    #print("Shape 'time_in_cycles_val':              ", time_in_cycles_val.shape)
    #print("Shape 'unit_val':                        ", unit_val.shape, end="\n\n")
    

    return X_train_norm, y_train, time_in_cycles_train, unit_train,\
        X_test_norm, y_test, time_in_cycles_test, unit_test,\
        X_val_norm, y_val, time_in_cycles_val, unit_val,\
        standardization_params


def pipeline_level_2(input_file,
                     subwindow_features,
                     min_window_length=100,
                     subwindows=32,
                     subwindow_length=50,
                     subwindow_target="max",
                     maximize_subwindows=False,
                     shuffle_subwindows=True,
                     percentage_training=70,
                     percentage_test=15,
                     splitting_feature="unit_number",
                     regression_feature="time_in_cycles",
                     target_feature_name="target",
                     target_unit=pd.Timedelta("1 h"),
                     target_clip_lower=False,
                     target_min=10,
                     target_clip_upper=False,
                     target_max=125,
                     fixed_dataframe_assignment=True,
                     sort_dataframes_after_splitting=False,
                     verbose=True):

    if percentage_test + percentage_test > 100:
        raise ValueError("More than 100 % of data can't be splitted")

    print("Chosen splitting [%]:")
    print("Training:          ", percentage_training)
    print("Test:              ", percentage_test)
    print("Validation:        ", 100 - (percentage_training + percentage_test))

    # Create list of dataframes
    list_of_dataframes = subdivide_dataframe_by_feature(input_file,
                                                        feature=splitting_feature)

    # Add target values to each dataframe
    for dataframe in list_of_dataframes:
        add_target_feature(dataframe,
                           regression_feature=regression_feature,
                           target_feature_name=target_feature_name,
                           unit=target_unit,
                           target_clip_lower=target_clip_lower,
                           target_min=target_min,
                           target_clip_upper=target_clip_upper,
                           target_max=target_max)

    subwindowing_kwargs = dict(window_length=min_window_length,
                               subwindows=subwindows,
                               subwindow_length=subwindow_length,
                               subwindow_target=subwindow_target,
                               maximize_subwindows=maximize_subwindows,
                               shuffle=shuffle_subwindows)

    all_time_in_cycles,\
        all_target_values = get_subwindows_from_dataframes(list_of_dataframes,
                                                           features=[regression_feature],
                                                           target_feature=target_feature_name,
                                                           **subwindowing_kwargs)

    all_time_in_cycles = np.concatenate(all_time_in_cycles).astype(int)

    all_feature_values,\
        all_unit_numbers = get_subwindows_from_dataframes(list_of_dataframes,
                                                          features=subwindow_features,
                                                          target_feature=splitting_feature,
                                                          **subwindowing_kwargs)

    all_unit_numbers = all_unit_numbers.astype(int)
    
    print("Number of dataframes:          ", len(list_of_dataframes))
    print("Shape 'all_feature_values':    ", all_feature_values.shape)
    print("Shape 'all_time_in_cycles':    ", all_unit_numbers.shape)
    print("Shape 'all_unit_numbers':      ", all_unit_numbers.shape)
    print("Shape 'all_unit_numbers':      ", all_target_values.shape)

    list_indices = list(range(all_feature_values.shape[0]))

    # Create lists to chose dataframes with corresponding index for:
    # - training   (-> first_list_indices)
    # - test       (-> second_list_indices)
    # - validation (-> third_list_indices)

    proportion_list_1 = percentage_training * 1e-2
    proportion_list_2 = percentage_test / (100 - percentage_training)

    first_list_indices,\
        second_list_indices = subdivide_indices_of_a_list(list_indices,
                                                          first_list_proportion=proportion_list_1,
                                                          fixed_seed=fixed_dataframe_assignment,
                                                          sort_lists=sort_dataframes_after_splitting)

    # print("'first_list_indices':                     ", first_list_indices)
    # print("'second_list_indices' (before splitting): ", second_list_indices)

    second_list_indices,\
        third_list_indices = subdivide_indices_of_a_list(second_list_indices,
                                                         first_list_proportion=proportion_list_2,
                                                         fixed_seed=fixed_dataframe_assignment,
                                                         sort_lists=sort_dataframes_after_splitting)

    # print("'second_list_indices' (after splitting):  ", second_list_indices)
    # print("'third_list_indices':                     ", third_list_indices)

    X_train, y_train = all_feature_values[first_list_indices], all_target_values[first_list_indices]
    X_test, y_test = all_feature_values[second_list_indices], all_target_values[second_list_indices]
    X_val, y_val = all_feature_values[third_list_indices], all_target_values[third_list_indices]

    time_in_cycles_train, unit_train = all_time_in_cycles[
        first_list_indices], all_unit_numbers[first_list_indices]
    time_in_cycles_test, unit_test = all_time_in_cycles[
        second_list_indices], all_unit_numbers[second_list_indices]
    time_in_cycles_val, unit_val = all_time_in_cycles[
        third_list_indices], all_unit_numbers[third_list_indices]

    # Normalization
    X_train_norm,\
        standardization_params = normalize_feature_values(
            features_array=X_train)

    X_test_norm, _ = normalize_feature_values(features_array=X_test,
                                              train_standardization_params=standardization_params)

    X_val_norm, _ = normalize_feature_values(features_array=X_val,
                                             train_standardization_params=standardization_params)

    print("Number of dataframes:                    ",
          len(list_of_dataframes), end="\n\n")
    print("Shape 'X_train' before normalization:    ", X_train.shape)
    print("Shape 'X_train' after normalization:     ", X_train_norm.shape)
    print("Shape 'y_train':                         ", y_train.shape)
    print("Shape 'time_in_cycles_train':            ", time_in_cycles_train.shape)
    print("Shape 'unit_train':                      ",
          unit_train.shape, end="\n\n")

    print("Shape 'X_test' before normalization:     ", X_test.shape)
    print("Shape 'X_test_norm' after normalization: ", X_test_norm.shape)
    print("Shape 'y_test':                          ", y_test.shape)
    print("Shape 'time_in_cycles_test':             ", time_in_cycles_test.shape)
    print("Shape 'unit_test':                       ",
          unit_test.shape, end="\n\n")

    print("Shape 'X_val' before normalization:      ", X_val.shape)
    print("Shape 'X_val_norm' after normalization:  ", X_val_norm.shape)
    print("Shape 'y_val':                           ", y_val.shape)
    print("Shape 'time_in_cycles_val':              ", time_in_cycles_val.shape)
    print("Shape 'unit_val':                        ", unit_val.shape, end="\n\n")

    return X_train_norm, y_train, time_in_cycles_train, unit_train,\
        X_test_norm, y_test, time_in_cycles_test, unit_test,\
        X_val_norm, y_val, time_in_cycles_val, unit_val,\
        standardization_params


if __name__ == "__main__":
    pass

