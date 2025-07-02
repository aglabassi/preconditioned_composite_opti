#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:25:42 2025

@author: aglabassi
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob

from matplotlib.lines import Line2D

def collect_compute_mean(keys, loss_ord, r_true, res, methods, problem, base_dir):
    losses = dict(( method, dict()) for method in methods )
    stds = dict( (method, dict()) for method in methods)
    times = dict( (method, dict()) for method in methods)

    for rank, cond_number in keys:
        for method in methods:
            file_pattern = os.path.join(base_dir, 
                                        f"{('res' if res else 'exp') + problem}_{method}_l_{loss_ord}_r*={r_true}_r={rank}_condn={cond_number}_trial_*.csv")
            file_list = glob.glob(file_pattern)
            data_list = []
            
            
            # Read each file and append its data to the data_list
            for file in file_list:
                data_list.append(np.loadtxt(file, delimiter=','))  # Assume the CSV is correctly formatted for np.loadtxt
                
            file_pattern_time = os.path.join(base_dir, 
                                        f"{('times') + problem}_{method}_l_{loss_ord}_r*={r_true}_r={rank}_condn={cond_number}_trial_*.csv")
            file_list_time = glob.glob(file_pattern_time)
            data_list_time = []
            
            
            # Read each file and append its data to the data_list
            for file in file_list_time:
                data_list_time.append(np.loadtxt(file, delimiter=','))  # Assume the CSV is correctly formatted for np.loadtxt
                
                
                
            
            # Convert the list of arrays into a 2D numpy array for easier manipulation
            data_array = np.array(data_list)
            time_array = np.array(data_list_time)
            
            # Compute the mean across all trials (rows) for each experiment
            mean_values = np.mean(data_array, axis=0)
            time_values = np.mean(time_array, axis=0)
            std = np.std(data_array, axis=0)
            
            losses[method][(rank, cond_number)]  = mean_values
            times[method][(rank, cond_number)] = time_values
            stds[method][(rank, cond_number )]  = std
            
   
    return losses, stds, times


def plot_losses_with_styles(losses, stds, r_true, loss_ord, base_dir, problem, kappa, num_dots=0.1, intro_plot=False, symmetric=True, 
                            tensor=False,had=False, d=None,loss2=None, stds2=None, times=None, noneg=True, fixed_ylim=False):
    """
    Plots the losses with distinct styles for methods, parameterizations, and ill-conditioning levels.

    Parameters:
    - losses (dict): Nested dictionary containing loss values.
    - stds (dict): Nested dictionary containing standard deviations of losses.
    - r_true (float): The true parameterization value for comparison.
    - loss_ord (int): An identifier for the loss order.
    - base_dir (str): Base directory to save the plot.
    - problem (str): Type of problem (e.g., 'Burer-Monteiro', 'Asymmetric', etc.).
    - kappa (float): Ill-conditioning parameter.
    - num_dots (float): Fraction to determine the number of points to plot.
    """

    # Enable LaTeX rendering if desired
    mpl.rcParams['text.usetex'] = True  # Set to True if LaTeX is installed and desired

    # Adjust matplotlib parameters for publication quality
    mpl.rcParams['figure.figsize'] = (12, 9)  # Increased size for better visibility
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math rendering
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['xtick.labelsize'] = 25
    mpl.rcParams['ytick.labelsize'] = 25
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['lines.linewidth'] = 5
    mpl.rcParams['lines.markersize'] = 10



    colors = dict()
    


    methods_ = list(losses.keys())
    
    for m in methods_:
        for m2 in methods_:
            assert losses[m].keys() == losses[m2].keys(), "All methods must have the same keys."



    markers = ['o', '^', 'D', 's', 'P', '*', 'X', 'v', '<', '>']  # Extended markers list
    linestyles = ['-', ':']  # Different linestyles

    if intro_plot:
        markers = ['o']*10
        linestyles = [(0, (1.5,1.3))]*10
        
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='major', labelsize=25)

    keys = list(losses[methods_[0]].keys())

    rs = []
    cs = []
    for k in keys:
        r, c = k
        if r not in rs:
            rs.append(r)
        if c not in cs:
            cs.append(c)

    # Machine epsilon for float64
    epsilon_machine = np.finfo(float).eps

    # Initialize dictionaries to store unique styles for legends
    marker_styles = {}
    linestyle_styles = {}
    
    method_colors = {
    'Polyak Subgradient': '#dc267f',
    'Gradient descent': '#dc267f',  # Same color as 'Polyak Subgradient'
    'Scaled gradient': '#ffb000',
    'Scaled gradient($\lambda=10^{-8}$)': '#ffb000',
    'Scaled subgradient': '#ffaf00',  # Same color as 'Scaled gradient'
    'OPSA($\lambda=10^{-8}$)': '#ffb000',
    'Precond. gradient': '#fe6100',
    'Gauss-Newton': '#648fff'  ,
    'Levenberg-Marquardt (ours)': '#785ef0',
    }

    methods = [ method for idx_m, (method, color) in enumerate(method_colors.items(), start=1)   if method in methods_ ]

    for idx_, method in enumerate(methods):

        color =  method_colors[method] 

        
        for idx, k in enumerate(keys):
            r_index = rs.index(k[0])
            c_index = cs.index(k[1])
            errs = np.array(losses[method][k])
            std = np.array(stds[method][k])

            # Determine the index where errors have converged to machine epsilon
            convergence_threshold = 1e-13 if not had else 1e-13  # Slightly above machine epsilon to account for numerical errors
            divergence_threshold  = 1e3
            converged_indices = np.where(errs <= convergence_threshold)[0]
            diverged_indices = np.where(errs  >= divergence_threshold)[0]
            if converged_indices.size > 0:
                last_index = converged_indices[0] + 1  # Include the converged point
            elif diverged_indices.size > 0:
                last_index = diverged_indices[0] + 1
            else:
                last_index = len(errs)

            # Slice the errors up to the convergence point
            errs = errs[:last_index]
            std = std[:last_index]

            num_dots_adapted = int(num_dots * last_index)
            if k[1] > 1:
                num_dots_adapted //=2

            start = 0
            num_dots_adapted = max(1, num_dots_adapted)
            indices = np.arange(start, last_index, num_dots_adapted)
            if k[1] > 1:
                indices = [ i for idx,i in enumerate(indices) if idx %2 != 0]

            indices = np.hstack((np.zeros(1, dtype=int), indices))
            if indices.size == 0:
                indices = np.array([0])  # Ensure at least one point is plotted
            elif indices[-1] != last_index - 1:
                # Include the last index to reach the convergence threshold
                indices = np.append(indices, last_index - 1)

            # Determine parameterization (overparam or exact param)
            over_exact_label = "Overparameterized" if k[0] > r_true else "Exact parameterization"
            if over_exact_label not in linestyle_styles:
                linestyle = linestyles[1 if k[0] > r_true else 0]
                linestyle_styles[over_exact_label] = linestyle  # Store linestyle for the parameterization
            else:
                linestyle = linestyle_styles[over_exact_label]

            # Determine marker based on ill-conditioning (kappa value)
            kappa_label = "Ill-conditioned" if k[1] > 1 else "Well-conditioned"
            if kappa_label not in marker_styles:
                marker = markers[1 if k[1] > 1 else 0]
                marker_styles[kappa_label] = marker  # Store marker for the kappa value
            else:
                marker = marker_styles[kappa_label]

            # -------------------- CHANGED LINES (add time support) --------------------
            indices = indices.astype(int)
            if times is not None:
                times_arr = np.array(times[method][k])[:last_index]
                x_vals = times_arr[indices]
            else:
                x_vals = indices
            # -------------------------------------------------------------------------
            indices = indices.astype(int)
            ax.plot(
                x_vals,
                errs[indices],
                linestyle=linestyle,
                color=color,
                marker=marker,
                label=None  # Labels are handled separately
            )

            # Fill between for error bands
            ax.fill_between(
                x_vals,
                errs[indices] - std[indices],
                errs[indices] + std[indices],
                alpha=0.2,
                color=color
            )
            
            if loss2 is not None:
                errs2 = np.array(loss2[method][k])
                converged_indices = np.where(errs2 <= convergence_threshold)[0]
                diverged_indices = np.where(errs2  >= divergence_threshold)[0]
                if converged_indices.size > 0:
                    last_index = converged_indices[0] + 1  # Include the converged point
                elif diverged_indices.size > 0:
                    last_index = diverged_indices[0] + 1
                else:
                    last_index = len(errs)
                
                # Slice the errors up to the convergence point
                errs = errs[:last_index]
                std = std[:last_index]
                
                num_dots_adapted = int(num_dots * last_index)
                if k[1] > 1:
                    num_dots_adapted //=2
                
                start = 0
                num_dots_adapted = max(1, num_dots_adapted)
                indices = np.arange(start, last_index, num_dots_adapted)
                if k[1] > 1:
                    indices = [ i for idx,i in enumerate(indices) if idx %2 != 0]
                
                indices = np.hstack((np.zeros(1, dtype=int), indices))
                if indices.size == 0:
                    indices = np.array([0])  # Ensure at least one point is plotted
                elif indices[-1] != last_index - 1:
                    # Include the last index to reach the convergence threshold
                    indices = np.append(indices, last_index - 1)
                    
                # --------------- CHANGED: pick x for second curve ----------------
                if times is not None:
                    times_arr = np.array(times[method][k])[:last_index]
                    x_vals2 = times_arr[indices]
                else:
                    x_vals2 = indices
                # -----------------------------------------------------------------

                ax.plot(x_vals2, errs2[indices],
                        linestyle=linestyle,
                        color=color,
                        marker=marker,
                        label=None,
                        alpha=0.5)  # ← faded more
            
                # optional: error band for loss2
                if stds2 is not None:
                    std2 = np.array(stds2[method][k])[:last_index]
                    ax.fill_between(x_vals2,
                                    errs2[indices] - std2[indices],
                                    errs2[indices] + std2[indices],
                                    alpha=0.1,  # even lighter
                                    color=color)


    # Set the plot title based on the problem type
    object_ = {
        'Burer-Monteiro': 'Matrix',
        'Asymmetric': 'Matrix',
        'Hadamard': 'Square Vector',
        'Tensor': 'Tensors'

    }.get(problem, '')

    # ------------------- CHANGED: conditional xlabel -------------------
    if times is not None:
        ax.set_xlabel('Time (s)', fontsize=30)
    else:
        ax.set_xlabel('Iteration k', fontsize=30)
    # -------------------------------------------------------------------

    metric = ('X_k' + ('X_k^{\\top}' if symmetric else 'Y_k^{\\top}') + ' - M^{\star}') if not tensor else ('F_{{\\mathrm{sym}}}({X_k}) - T^\star' if symmetric else 'F(W_k,X_k,Y_k) - T^\star ')
    denominator = 'M^\star' if not tensor else 'T^\star'
    ax.set_ylabel(fr'Relative Distance $\frac{{\left\| {metric} \right\|_F}}{{\left\| {denominator} \right\|_F}}$', fontsize=30)
    if had:
        if noneg:
            ax.set_ylabel(fr'Relative Distance $\frac{{\left\| x_k \odot x_k - z^\star \right\|_2}}{{\left\| z^\star \right\|_2}}$', fontsize=30)
        else:
            ax.set_ylabel(fr'Relative Distance $\frac{{\left\| x_k  - z^\star \right\|_2}}{{\left\| z^\star \right\|_2}}$', fontsize=30)

    if intro_plot:
        ax.set_ylabel(r'Relative Distance $\frac{\left\| M_k - M^\star \right\|_F}{\left\| M^\ast \right\|_F}$', fontsize=30)
        
    ax.set_yscale('log')
    if fixed_ylim:                     # turn it on/off from the caller
        ax.set_ylim(1e-14, 1e2)        # bottom, top (log-scale OK)

    # Adding grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjusting tick parameters
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1, labelsize=26)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.5, labelsize=26)

    # Create custom legend handles
    # Methods legend (colors only)
    if intro_plot:
        method_handles = [
            Line2D([0], [0], color=method_colors[method], linestyle=linestyle, marker=marker, lw=3) for method in methods
        ]
    else:
        method_handles = [
            Line2D([0], [0], color=method_colors[method], lw=2) for method in methods
        ]
        
    method_labels = []
    for method in methods:
        method_labels.append(method if method != 'Levenberg-Marquardt (ours)' else 'LMM (ours)')

    # Markers legend (parameterization), markers in black
    marker_handles = [
        Line2D([0], [0], marker=marker_styles[label], color='black', linestyle='None', markersize=8)
        for label in marker_styles
    ]
    marker_labels = list(marker_styles.keys())

    # Linestyles legend (ill-conditioning), linestyles in black
    linestyle_handles = [
        Line2D([0], [0], linestyle=linestyle_styles[label], color='black', lw=2)
        for label in linestyle_styles
    ]
    linestyle_labels = list(linestyle_styles.keys())
    # Combined legend for markers and linestyles at upper right
    combined_handles = marker_handles + linestyle_handles
    combined_labels = marker_labels + linestyle_labels

    # Determine the number of entries in each legend
    num_legend1 = len(method_handles)
    num_legend2 = len(marker_handles) + len(linestyle_handles)

    # Find the maximum number of entries
    max_entries = max(num_legend1, num_legend2)

    # Function to create a dummy handle
    def create_dummy_handle():
        return Line2D([0], [0], color='white', lw=0, markersize=0, label='_nolegend_')

    # Add dummy handles to legend1 if it's shorter
    if num_legend1 < max_entries:
        num_dummies = max_entries - num_legend1
        for _ in range(num_dummies):
            method_handles.append(create_dummy_handle())
            method_labels.append('')  # Empty label for dummy handle

    # Add dummy handles to legend2 if it's shorter
    if num_legend2 < max_entries:
        num_dummies = max_entries - num_legend2
        for _ in range(num_dummies):
            combined_handles.append(create_dummy_handle())
            combined_labels.append('')  # Empty label for dummy handle


    height = 0.97


    # Methods legend at upper right
    legend1 = ax.legend(
        method_handles,
        method_labels,
        title='Methods',
        loc='upper right',
        bbox_to_anchor=(0.67 if not intro_plot else 0.95, height),
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=18 if not intro_plot else 30,          # Set the font size of the legend labels to 18
        title_fontsize=20 if not intro_plot else 30     # Set the font size of the legend title to 20
    )


    if not intro_plot:
        legend2 = ax.legend(
            combined_handles,
            combined_labels,
            title='Setting',
            loc='upper right',
            bbox_to_anchor=(1, height),  # Adjust the y-coordinate as needed
            frameon=True,
            facecolor='white',
            edgecolor='black',
            fontsize=18,          # Set the font size of the legend labels to 18
            title_fontsize=20     # Set the font size of the legend title to 20
        )
        ax.add_artist(legend2)

    # Add the first legend back to the axes
    ax.add_artist(legend1)

    # Tight layout for better spacing
    plt.tight_layout()

    # Saving the figure in a vector format (PDF)
    fig_path = os.path.join(base_dir, f'./{"intro_" if intro_plot else ""}exp_l{loss_ord}_{problem}_{d}_{"iteration" if times is None else "time"}.pdf')
    plt.savefig(fig_path, format='pdf', bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_transition_heatmap(
    success_matrixes: dict,
    d_trials: list,
    n: int,
    base_dir: str,
    keys: tuple,
    problem: str = 'TransitionPlot',
    max_corr: float = 0.5,
):
    """
    Plots one heatmap for each method stored in success_matrixes 
    (keys are method names, values are success_matrix).
    All subplots share the same color scale (Success Ratio in [0,1])
    and a single colorbar.
    """

    font_size = 30
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size

    # Adjust the figure size as desired:
    num_methods = len(success_matrixes)
    fig, axs = plt.subplots(
        num_methods, 1,
        figsize=(12, 6*num_methods),  # scale width by #methods
        dpi=300,
        squeeze=False  # so axs is always 2D: shape (1, num_methods)
    )

    # For consistent color scale across subplots:
    vmin, vmax = 0, 1

    # We'll keep track of the mappable (the last image) to create the colorbar
    im = None

    # Plot each method's heatmap in its own row
    methods = success_matrixes
    for i, method in enumerate(methods):
        ax = axs[i, 0]
        success_matrix = success_matrixes[method]

        # Show the heatmap
        im = ax.imshow(
            success_matrix,
            cmap='Greys_r',
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            vmin=vmin,
            vmax=vmax,
            extent=(0, max_corr, 0, len(d_trials))
        )

        # Major x-ticks at [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        major_ticks_x = np.arange(0, max_corr + 0.01, 0.1)
        ax.set_xticks(major_ticks_x)
        ax.set_xticklabels([f"{val:.1f}" for val in major_ticks_x])
        ax.invert_yaxis()

        # Minor x-ticks every 0.02, excluding major ticks
        all_ticks_x = np.arange(0, max_corr + 0.025, 0.025)
        minor_ticks_x = [t for t in all_ticks_x if t not in major_ticks_x]
        ax.set_xticks(minor_ticks_x, minor=True)

        # Major y-ticks at 1, 3, 5, 7, ... (starting at 1)
        major_ticks_y = list(range(0, len(d_trials), 2))
        ax.set_yticks(major_ticks_y)
        ax.set_yticklabels([f"{d_trials[t] / (2 * n):.0f}" for t in major_ticks_y])

        # Minor y-ticks at 2, 4, 6, 8, ... (without labels)
        minor_ticks_y = list(range(1, len(d_trials), 2))
        ax.set_yticks(minor_ticks_y, minor=True)

        # Label only the leftmost subplot's y-axis to avoid duplication:
        if i == num_methods - 1:
            ax.set_xlabel(r"Corruption Level", fontsize=font_size)

        if i == num_methods // 2:
            ax.set_ylabel(r"Measurement Ratio $m / (2n)$", fontsize=font_size)

        # Give each subplot a title corresponding to its method:
        #ax.set_title(method, fontsize=font_size)

    # Create one colorbar for all subplots, using the last im
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.set_ylabel("Success Ratio", fontsize=font_size*1.2)
    cbar.ax.tick_params(labelsize=font_size)

    # Save the figure
    save_path = os.path.join(base_dir, f"{problem}_{keys}_transition_plot.pdf")
    plt.savefig(save_path, format='pdf')
    print(f"Figure saved to: {save_path}")
    plt.show()
    
def plot_results_sensitivity(to_be_plotted, corr_level, q, r_test, c, 
                             gammas, lambdas, font_size, rel_error_exp, problem,
                             base_dir, plot_x=False, plot_y=False):
    # LaTeX / font settings
    font_size = 30
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.size'] = font_size
    


    method_colors = {
        'Polyak Subgradient': '#dc267f',
        'Gradient descent': '#dc267f',  # Same color as 'Polyak Subgradient'
        'Scaled gradient': '#ffb000',
        'Scaled gradient($\lambda=10^{-8}$)':  '#ffaf00',  
        'Scaled subgradient': '#ffaf00',  # Same color as 'Scaled gradient'
        'OPSA($\lambda=10^{-8}$)': '#ffaf00',  
        'Precond. gradient': '#fe6100',
        'Gauss-Newton': '#648fff',
        'Levenberg-Marquardt (ours)': '#785ef0'
    }

    for i, lambda_ in enumerate(lambdas):
        plt.figure(figsize=(10, 6))  # Create a new figure for each lambda
        for method in to_be_plotted.keys():
            color = method_colors[method]
            data = [to_be_plotted[method][i, j][0] for j in range(len(gammas))]
            noise  = [to_be_plotted[method][i, j][1] for j in range(len(gammas)) ]  # Assume noise is a tuple (low, high)
    
            # Compute the noise bounds
            noise_low = [noise[j][0] for j in range(len(gammas))]
            noise_high = [noise[j][1] for j in range(len(gammas))]
    
            # Plot scatter points, connecting lines, and noise shading
            plt.plot(gammas, data, label=(method if method != 'Levenberg-Marquardt (ours)' else 'LMM (ours)'), color=color, linestyle='-', linewidth=2, marker='o')
            plt.fill_between(gammas, noise_low, noise_high, color=color, alpha=0.2)
    
        # Set labels and title with LaTeX rendering
        
        plt.xlabel(r"Hyperparameter $\gamma$", color='black' if plot_x else 'white', fontsize=font_size)
        bound = 10
        
        plt.ylabel(rf"Iterations to convergence", color='black' if plot_y else 'white', fontsize=font_size)
        #plt.title(f"$q={q}$", fontsize=font_size) 
        plt.xscale('log')
        plt.xticks(gammas, fontsize=font_size//2)  # Explicitly set ticks and labels
        
        # Customize ticks and add grid
        plt.xticks(fontsize=font_size//2)
        plt.yticks(fontsize=font_size//2)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=font_size//2, loc="lower right")
    
        # Save the plot to a file
        save_path = os.path.join(base_dir, f"plot_{problem}_{q}_{corr_level}_{r_test}_{c}_{lambda_}.pdf")
        plt.savefig(save_path, format='pdf')
        print(f"Figure saved to: {save_path}")
        plt.show()
        plt.close()  # Close the figure to free memory
        
    





