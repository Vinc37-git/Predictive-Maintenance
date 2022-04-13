# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:13:55 2021

@author: Fabian Moeller and Vincent Hackstein
"""

from datetime import datetime
import math

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from tqdm import tqdm

from data_preprocessing import subdivide_dataframe_by_feature
from regression_models import get_rul_score


def return_current_date_and_time():
    '''
    Function to get the current data and time as a string.
    Used to save figures with their timestamp of creation.

    Returns
    -------
    current_date : str
        Current date as a string
    current_time : str
        Current time as a string

    '''
    current_date = str(datetime.now().strftime("%d-%m-%Y"))
    current_time = str(datetime.now().strftime("%H-%M-%S")+"_Uhr")
    return current_date, current_time


def save_figure(saving_path,
                figure_name,
                save_fig=True,
                plot_format="pdf"):
    '''
    Saving figure; therefore remove invalid symbols in filenames and replace them with an underscore

    Parameters
    ----------
    saving_path : str
        DESCRIPTION.
    figure_name : str
        DESCRIPTION.
    save_fig : bool, optional
        True: plot will be saved. False: plot won't be saved. The default is True.
    plot_format : str, optional
        Either "pdf" or "png" or other picture-format. The default is "pdf".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if save_fig is True:
        current_date, current_time = return_current_date_and_time()
        filename = "".join([x if x.isalnum() else "_" for x in figure_name])
        fig = plt.gcf()
        plt.tight_layout()
        fig.savefig(saving_path + "/" + current_date + "_" + current_time +
                    "_" + filename + "." + plot_format)
    elif save_fig is False:
        print("Figure not saved")
    else:
        raise ValueError("%s not allowed in save_fig" % save_fig)


def truncate_colormap(cmap, minimum=0, maximum=1, number_of_colors=100):
    '''
    Function to cut off specific parts of a colormap.
    Can be used to take only colors from green to red in plt.cm.jet

    Parameters
    ----------
    cmap : TYPE
        DESCRIPTION.
    minimum : TYPE, optional
        DESCRIPTION. The default is 0.
    maximum : TYPE, optional
        DESCRIPTION. The default is 1.
    number_of_colors : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    new_cmap : TYPE
        DESCRIPTION.

    '''
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minimum, b=maximum),
        cmap(np.linspace(minimum, maximum, number_of_colors)))
    return new_cmap


def plot_correlation_heatmap(input_file,
                             saving_path="",
                             save_fig=False,
                             plot_format="pdf"):

    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(input_file.corr()))
    heatmap = sns.heatmap(input_file.corr(),
                          mask=mask,
                          vmin=-1, vmax=1,
                          annot=False,
                          cmap=plt.cm.hsv,
                          xticklabels=True,
                          yticklabels=True,
                          square=True)

    save_figure(saving_path=saving_path,
                figure_name="Correlation_heatmap",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_percentage_of_removed_cycles(input_file_rul,
                                      input_file_test,
                                      saving_path="",
                                      save_fig=False,
                                      plot_format="pdf"):

    total_cycles = []
    for _, data in input_file_test.groupby("unit_number"):
        total_cycles.append(data["time_in_cycles"].max())

    x_values = range(len(input_file_rul))
    y_values = 100 * input_file_rul.values / np.array(total_cycles)
    #with plt.xkcd():
    plt.figure(figsize=(10, 10))
    plt.bar(x=x_values, height=y_values);
    plt.plot(x_values, len(x_values) * [np.mean(y_values)], c="r", label="mean")
    plt.xlabel("Engine no.")
    plt.ylabel("Ratio of RUL to max(time_in_cycles)")
    plt.legend(loc="upper right")
    plt.title("Percentage of removed cycles in test dataset")

    save_figure(saving_path=saving_path,
                figure_name="Ratio_cycles_in_test_to_rul_dataset",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_samples_in_test_and_rul_dataset(input_file_test,
                                         input_file_rul,
                                         saving_path="",
                                         save_fig=False,
                                         plot_format="pdf"):

    list_of_dataframes = subdivide_dataframe_by_feature(input_file_test,
                                                        feature="unit_number")

    x_values = input_file_test["unit_number"].unique()
    y_values = [len(df) for df in list_of_dataframes]

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    bar_kwargs = dict(x=x_values,
                      width=1,
                      edgecolor="k")
    ax.bar(height=y_values, label="Measured cycles", **bar_kwargs)
    ax.bar(height=input_file_rul, bottom=y_values,
           color="blue", alpha=0.5, label="RUL", **bar_kwargs)
    ax.set_xlabel("Units")
    ax.set_ylabel("Time in cycles")

    ax.plot(x_values, [np.mean(y_values)] * len(x_values), color="orange", label="Mean all measured cycles", lw=3)
    ax.plot(x_values, [np.min(y_values)]  * len(x_values), color="r", label="Min all measured cycles", lw=3)
    ax.legend(ncol=2)
    ax.set_title("Cycles in 'test' and 'RUL' dataset")
    save_figure(saving_path=saving_path,
                figure_name="Cycles_in_test_and_rul_dataset",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_colorbar(cmap,
                  cmap_ticks,
                  cmap_ticklabels,
                  title,
                  figure_size=(10, 2),
                  saving_path="",
                  save_fig=False,
                  plot_format="pdf"):

    fig, ax = plt.subplots(figsize=figure_size)

    norm = mpl.colors.Normalize(vmin=min(cmap_ticks),
                                vmax=max(cmap_ticks))
    custom_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(custom_map, cax=ax,
                            extend="both",
                            orientation="horizontal")

    colorbar.set_ticks(cmap_ticks)
    colorbar.set_ticklabels(cmap_ticklabels)

    ax.set_title(label=title, pad=20, weight="bold")
    save_figure(saving_path=saving_path,
                figure_name="Colormap" + title,
                save_fig=save_fig,
                plot_format=plot_format)


def plot_mse_and_rul_scoring_function(saving_path="",
                                      save_fig=False,
                                      plot_format="pdf"):

    x_values = np.linspace(-40, 40, 1000)
    ground_truth = [0] * len(x_values)

    #with plt.xkcd():
    plt.figure(figsize=(10, 8))
    plt.plot(x_values, get_rul_score(ground_truth, x_values, return_array=True),
             label="RUL-SCORE")
    plt.plot(x_values, np.abs(x_values),
             c="r", label="RMSE")
    plt.xlabel("Difference between prediction and ground truth\t" +
               "($ h_{i} = \mathit{RUL}_{pred, i} - \mathit{RUL}_{gt, i}$)")
    plt.ylabel("RUL-SCORE / MSE")
    plt.legend(loc="upper left", ncol=1)
    plt.title("Comparision of RUL Scoring Function and RMSE")

    save_figure(saving_path=saving_path,
                figure_name="RUL_scoring_function_with_mse",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_samples_over_feature(initial_dataframe,
                              feature="unit_number",
                              saving_path="",
                              save_fig=True,
                              plot_format="pdf"):

    list_of_dataframes = subdivide_dataframe_by_feature(initial_dataframe=initial_dataframe,
                                                        feature=feature,
                                                        drop_feature=False)

    length_of_dataframes = []
    for dataframe in list_of_dataframes:
        length_of_dataframes.append(len(dataframe))

    # Plot histogram
    unique_values = initial_dataframe[feature].unique()
    len_ave = np.mean(length_of_dataframes)
    len_min = min(length_of_dataframes)
    len_max = max(length_of_dataframes)

    fig = plt.figure(figsize=(10, 10))
    axis = plt.subplot()
    axis.bar(unique_values, length_of_dataframes)
    axis.plot(unique_values, [len_ave] * len(length_of_dataframes), c="g", label="Average: %s" % str(len_ave).replace(".", ","))
    axis.plot(unique_values, [len_min] * len(length_of_dataframes), c="k", label="Min:     %s" % len_min)
    axis.plot(unique_values, [len_max] * len(length_of_dataframes), c="r", label="Max:     %s" % len_max)
    axis.set_xticks(np.linspace(min(unique_values), max(unique_values), 10, endpoint=True, dtype=int))
    axis.legend(loc="lower right", framealpha=1)
    axis.set_title("Histogram for feature: %s" % feature, weight="bold")
    axis.set_xlabel(r"%s" % feature)
    axis.set_ylabel(r"Samples per %s" % feature)

    figure_name = "Histogram_feature_" + feature
    save_figure(saving_path,
                figure_name,
                save_fig,
                plot_format)


def plot_all_features_for_all_units(list_of_dataframes,
                                    features,
                                    colormap_feature=None,
                                    number_of_columns=4,
                                    saving_path="",
                                    save_fig=False,
                                    plot_format="pdf"):

    rows = math.ceil(len(features) / number_of_columns)

    fig = plt.figure(figsize=(6 * number_of_columns, 6 * rows))  # in inches
    g = gridspec.GridSpec(rows, number_of_columns)#, wspace=0.01, hspace=0.01)


    x_feature = "time_in_cycles"

    for sub_df in tqdm(list_of_dataframes):
        x_values = sub_df[x_feature]
        for index, feature in enumerate(features):
            row_index = index // number_of_columns
            col_index = index % number_of_columns

            # Plot
            y_values = sub_df[feature]

            # define
            if colormap_feature is None:
                cm_feature = feature
            else:
                cm_feature = colormap_feature

            colormap_values = sub_df[cm_feature]
            norm = mpl.colors.Normalize(vmin=colormap_values.min(),
                                        vmax=colormap_values.max())

            axis = plt.subplot(g[row_index, col_index])
            axis.scatter(x_values.values[:-1], y_values.values[:-1],
                         c=norm(colormap_values[:-1]), cmap=plt.cm.rainbow, marker=".", zorder=2)
            axis.scatter(x_values.values[-1], y_values.values[-1], c="k", marker="*", zorder=3)
            axis.set_xlabel(x_feature)
            axis.set_ylabel(feature)
            axis.set_title("Sensor No. %s  |  Feature: %s" % (str(index + 1), feature),
                           {'fontweight': "bold"}, fontsize="medium")

    figure_name = "All_features_all_units_over_time" + "_cmap_" + str(cm_feature)
    save_figure(saving_path,
                figure_name,
                save_fig,
                plot_format)


def plot_signal_contributions_of_pcs(coefficients,
                                     features,
                                     pc_numbers=5,
                                     space_between_pcs=4,
                                     saving_path="",
                                     save_fig=False,
                                     plot_format="pdf"):
    '''

    Parameters
    ----------
    coefficients : TYPE
        DESCRIPTION.
    features : TYPE
        DESCRIPTION.
    pc_numbers : TYPE, optional
        DESCRIPTION: Number of principal components to be shown. The default is 5.
    space_between_pcs : TYPE, optional
        DESCRIPTION: Adjust spacing between single pcs. The default is 4.
    saving_path : TYPE, optional
        DESCRIPTION. The default is "".
    save_fig : TYPE, optional
        DESCRIPTION. The default is False.
    plot_format : TYPE, optional
        DESCRIPTION. The default is "pdf".

    Returns
    -------
    None.

    '''
    FONTSIZE = 20

    # Define color per principal component
    norm = mpl.colors.Normalize(vmin=0, vmax=pc_numbers - 1)
    cmap = plt.cm.gist_rainbow

    # Weighted coefficients
    wcoeff = 100 * abs(coefficients[:, :pc_numbers]) / np.sum(abs(coefficients[:, :pc_numbers]), axis=0)

    plt.figure(figsize=(15, 8))
    axis = plt.subplot()

    # For legend: Append only first entry of each principal component to handles
    handles = []
    for pc in range(pc_numbers):
        flag = True
        for feature_index in range(len(features)):
            x_value = pc + (pc_numbers + space_between_pcs) * feature_index
            y_value = wcoeff[feature_index, pc]
            bar = axis.bar(x_value, y_value, color=cmap(norm(pc)),
                           label="PC " + str(pc + 1), alpha=0.9)
            if flag:
                handles.append(bar)
                flag = False

    # Adjust legend
    n_cols = pc_numbers if pc_numbers <= 6 else 6
    axis.legend(handles=handles, ncol=n_cols,
                fontsize=FONTSIZE)

    # Adjust x- and y-ticks
    x_ticks = [x * (pc_numbers + space_between_pcs) + (pc_numbers - 1) / 2 for x in range(len(features))]
    axis.set_xticks(x_ticks)
    axis.set_xticklabels(features, rotation=90)

    # Adjust y-label
    axis.set_ylabel("Feature share in pc" + r" $\left[ \mathrm{\%} \right]$",
                    fontsize=FONTSIZE)

    plt.tight_layout(pad=5)

    save_figure(saving_path=saving_path,
                figure_name="Feature_share_in_%s_principal_compontents" % str(pc_numbers),
                save_fig=save_fig,
                plot_format=plot_format)


def plot_signal_contribution_for_single_pc(coefficients,
                                           features,
                                           pc_numbers=5,
                                           space_between_pcs=4,
                                           same_ylim=False,
                                           saving_path="",
                                           save_fig=False,
                                           plot_format="pdf"):

    # Weighted coefficients
    wcoeff = 100 * abs(coefficients[:, :pc_numbers]) /\
        np.sum(abs(coefficients[:, :pc_numbers]), axis=0)

    # Adjust spacing between single pcs
    space_between_features = 6

    # Define color per principal component
    norm = mpl.colors.Normalize(vmin=0, vmax=len(features) - 1)
    cmap = plt.cm.gist_rainbow

    fig = plt.figure(figsize=(15, 5 * pc_numbers))  # in inches
    g = gridspec.GridSpec(pc_numbers, 1)

    for pc in range(pc_numbers):
        axis = plt.subplot(g[pc, 0])
        x_ticks = []
        for feature_index in range(len(features)):
            x_value = feature_index * (1 + space_between_features)
            x_ticks.append(x_value)
            y_value = wcoeff[feature_index, pc]
            axis.bar(x_value, y_value, color=cmap(norm(feature_index)), alpha=0.9)
            axis.set_title("PC " + str(pc + 1), weight="bold")

        # Adjust x-ticks
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(features, rotation=90)

        # Adjust y-label
        axis.set_ylabel("Feature share in pc" + r" $\left[ \mathrm{\%} \right]$")

    if same_ylim:
        # Get y-limits for all axes
        y_lim_min = []
        y_lim_max = []
        for axis in fig.axes:
            y_lim_min.append(axis.get_ylim()[0])
            y_lim_max.append(axis.get_ylim()[1])

        # Set y-limits globally for all axes
        global_y_lim = [min(y_lim_min), max(y_lim_max)]
        for axis in fig.axes:
            axis.set_ylim(global_y_lim)

    plt.tight_layout(pad=5)
    save_figure(saving_path=saving_path,
                figure_name="Feature_share_for_%s_single_principal_compontents" % str(pc_numbers),
                save_fig=save_fig,
                plot_format=plot_format)




def plot_retained_variance(variance_ratio,
                           saving_path="",
                           save_fig=False,
                           plot_format="pdf"):

    fig = plt.figure(figsize=(10, 5))

    # Cumulative
    variances_cm = np.cumsum(variance_ratio)

    plt.plot(np.arange(1, len(variances_cm) + 1), 100 * variances_cm, marker = "o")
    plt.plot(np.arange(1, len(variances_cm) + 1), 100 * variance_ratio, marker = "o")

    FONTSIZE = 20
    PAD = 20
    plt.title("Retained variances for all pricipal components", weight="bold", fontsize=FONTSIZE, pad=PAD)
    plt.xlabel("Number of Principal Components", fontsize=FONTSIZE, labelpad=PAD)
    plt.ylabel("Retained Variance" + r" $\left[ \mathrm{\%} \right]$", fontsize=FONTSIZE, labelpad=PAD)
    plt.xticks(np.arange(1, len(variances_cm) + 1))
    plt.grid()
    plt.legend(["Cumulative", "Individual"], loc="best", fontsize=FONTSIZE)
    save_figure(saving_path=saving_path,
                figure_name="Retained_variances_pca",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_categories_over_pc_1_and_2(principal_component_df,
                                    urgency_indicators,
                                    urgency_series,
                                    saving_path="",
                                    save_fig=False,
                                    plot_format="pdf"):

    FONTSIZE = 20
    PAD = 20

    norm = mpl.colors.Normalize(vmin=0,
                                vmax=len(urgency_indicators) - 1)

    # Take only colors from green to red. Therefore start in mid of colormap
    cmap = truncate_colormap(cmap=plt.cm.jet,
                             minimum=0.5)

    plt.figure(figsize=(15, 8))
    axis = plt.subplot()


    # Start with 'urgent', since scatters are overlaying
    for index, indicator in enumerate(urgency_indicators[::-1]):
        sub_df = principal_component_df[urgency_series == indicator]
        axis.scatter(sub_df["PC_1"], sub_df["PC_2"], label=indicator,
                     color=cmap(norm(urgency_indicators.index(indicator))),
                     alpha=0.5)

    # Adjust labels
    axis.set_xlabel("PC 1", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_ylabel("PC 2", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_title("Categories over PC 1 and PC 2", weight="bold", fontsize=FONTSIZE, pad=PAD)
    axis.legend(fontsize=FONTSIZE)

    save_figure(saving_path=saving_path,
                figure_name="Class_distribution_pc_1_and_2",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_different_categories_for_one_feature(selected_df,
                                              feature,
                                              urgency_indicators,
                                              regression_feature="time_in_cycles",
                                              saving_path="",
                                              save_fig=False,
                                              plot_format="pdf"):

    x_values = selected_df[regression_feature]
    y_values = selected_df[feature]
    urgency_series = selected_df["Urgency_Indicator"]

    norm_1 = mpl.colors.Normalize(vmin=min(x_values),
                                  vmax=max(x_values))

    norm_2 = mpl.colors.Normalize(vmin=0,
                                  vmax=len(urgency_indicators) - 1)

    # Take only colors from green to red. Therefore start in mid of colormap
    cmap = truncate_colormap(cmap=plt.cm.jet,
                             minimum=0.5)

    fig = plt.figure(figsize=(12, 12))
    g = gridspec.GridSpec(2, 1)

    axis_0 = plt.subplot(g[0, 0])
    axis_1 = plt.subplot(g[1, 0])

    kwargs = dict(fontsize=20)
    axis_0.scatter(x_values, y_values, color=cmap(norm_1(x_values)))
    axis_0.set_title("Corrected core speed over cycles", weight="bold", pad=20, **kwargs)
    #axis_0.set_xlabel("Time [cycles]", labelpad=20, **kwargs)
    axis_0.set_ylabel("Corrected core speed [rpm]", labelpad=20, **kwargs)
    axis_0.set_xticklabels([])

    for index, indicator in enumerate(urgency_indicators):
        is_current_indicator = urgency_series == indicator
        axis_1.scatter(x_values[is_current_indicator],
                       y_values[is_current_indicator],
                       color=cmap(norm_2(index)))

    #axis_1.set_title("Corrected core speed over cycles", weight="bold", pad=20, **kwargs)
    axis_1.set_xlabel("Time [cycles]", labelpad=20, **kwargs)
    axis_1.set_ylabel("Corrected core speed [rpm]", labelpad=20, **kwargs)

    plt.tight_layout(pad=5)
    save_figure(saving_path=saving_path,
                figure_name="Demo_four_categories_over_time",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_binary_classification_over_pc_1_and_2(principal_component_df,
                                               urgency_series,
                                               saving_path="",
                                               save_fig=False,
                                               plot_format="pdf"):

    FONTSIZE = 20
    PAD = 20

    # Take only colors from green to red. Therefore start in mid of colormap
    norm = mpl.colors.Normalize(vmin=- 1, vmax=1)
    cmap = plt.cm.jet

    plt.figure(figsize =(15, 8))
    axis = plt.subplot()

    # Start with 'urgent', since scatters are overlaying
    is_urgent = urgency_series == "Urgent"

    axis.scatter(principal_component_df[is_urgent]["PC_1"],
                 principal_component_df[is_urgent]["PC_2"],
                 label="Urgent", color=cmap(norm(1)), alpha=0.5)
    axis.scatter(principal_component_df[~is_urgent]["PC_1"],
                 principal_component_df[~is_urgent]["PC_2"],
                 label="Not urgent", color=cmap(norm(0)), alpha=0.5)

    # Adjust labels
    axis.set_xlabel("PC 1", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_ylabel("PC 2", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_title("Binary class distribution over PC 1 and PC 2",
                   weight="bold", fontsize=FONTSIZE, pad=PAD)
    axis.legend(fontsize=FONTSIZE)
    save_figure(saving_path=saving_path,
                figure_name="Binary_classification_over_pc_1_and_2",
                save_fig=save_fig,
                plot_format=plot_format)


def plot_operational_settings_over_pc_1_and_2():
    urgency_indicators = ["Long", "Medium", "Short", "Urgent"]


    norm = mpl.colors.Normalize(vmin=0,
                                vmax=len(urgency_indicators) - 1)

    # Take only colors from green to red. Therefore start in mid of colormap
    cmap = truncate_colormap(cmap=plt.cm.jet,
                             minimum=0.5)

    plt.figure(figsize=(15, 8))
    axis = plt.subplot(projection="3d")

    markers = ["$1$", "$2$", "$3$"] # ["$\\alpha$", "$\\beta$", "$\gamma$"]
    
    # Start with 'urgent', since scatters are overlaying
    for index, indicator in enumerate(urgency_indicators[::-1]):
        sub_df = pc_df[urgency_series == indicator]
        x_values = sub_df["PC_1"]
        y_values = sub_df["PC_2"]
        for marker, setting in zip(markers, range(1, 4)):
            operational_setting = "operational_setting_" + str(setting)
            z_values = expanded_df[operational_setting][urgency_series == indicator]
            axis.scatter(x_values, y_values, z_values,
                         label=indicator, cmap=cmap, marker=marker,
                         color=cmap(norm(urgency_indicators.index(indicator))),
                         alpha=0.5,
                         s=40)
    
    # Adjust labels
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    axis.set_zlabel("Operation Settings")
    axis.set_title("Class distribution over PC 1 and PC 2", weight="bold")
    axis.legend();



def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_model_results(principal_component_df,
                       urgency_series,
                       X_train,
                       y_train,
                       y_pred,
                       model,
                       saving_path="",
                       save_fig=False,
                       plot_format="pdf"):

    FONTSIZE = 20
    PAD = 20

    # Take only colors from green to red. Therefore start in mid of colormap
    norm = mpl.colors.Normalize(vmin=- 1, vmax=1)
    cmap = plt.cm.jet

    plt.figure(figsize=(10, 10))
    axis = plt.subplot()

    # Set-up grid for plotting.
    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(axis, model, xx, yy, cmap=cmap, alpha=0.2)

    # Start with 'urgent', since scatters are overlaying
    is_urgent = urgency_series == "Urgent"
    is_correctly_predicted = y_train == y_pred

    # Urgent and correctly predicted
    is_correct_and_urgent = np.logical_and(is_urgent, is_correctly_predicted)

    axis.scatter(principal_component_df[is_correct_and_urgent]["PC_1"],
                 principal_component_df[is_correct_and_urgent]["PC_2"],
                 label="Urgent (True prediction): %s" % str(sum(is_correct_and_urgent)),
                 color=cmap(norm(1)), alpha=0.5)

    # Not urgent and correctly predicted
    is_correct_and_not_urgent = np.logical_and(~is_urgent, is_correctly_predicted)

    axis.scatter(principal_component_df[is_correct_and_not_urgent]["PC_1"],
                 principal_component_df[is_correct_and_not_urgent]["PC_2"],
                 label="Not urgent (True prediction): %s" % str(sum(is_correct_and_not_urgent)),
                 color=cmap(norm(0)), alpha=0.5)

    # Urgent and falsely predicted
    is_incorrect_and_urgent = np.logical_and(is_urgent, ~is_correctly_predicted)

    axis.scatter(principal_component_df[is_incorrect_and_urgent]["PC_1"],
                 principal_component_df[is_incorrect_and_urgent]["PC_2"],
                 label="Urgent (False prediction): %s" % str(sum(is_incorrect_and_urgent)),
                 color="magenta", marker="*", alpha=0.5)

    # Not urgent and falsely predicted
    is_incorrect_and_not_urgent = np.logical_and(~is_urgent, ~is_correctly_predicted)

    axis.scatter(principal_component_df[is_incorrect_and_not_urgent]["PC_1"],
                 principal_component_df[is_incorrect_and_not_urgent]["PC_2"],
                 label="Not urgent (False prediction): %s" % str(sum(is_incorrect_and_not_urgent)),
                 color="blue", marker="P", alpha=0.5)

    # Adjust labels
    axis.set_xlabel("PC 1", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_ylabel("PC 2", fontsize=FONTSIZE, labelpad=PAD)
    axis.set_title("Binary class distribution over PC 1 and PC 2",
                   weight="bold", fontsize=FONTSIZE, pad=PAD)
    axis.legend(fontsize=FONTSIZE)

    save_figure(saving_path=saving_path,
                figure_name="Model_results_binary_classification_over_pc_1_and_2",
                save_fig=save_fig,
                plot_format=plot_format)


if __name__ == "__main__":
    pass
