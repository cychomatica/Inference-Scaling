import json
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import argparse


def compare_results(file_basename, results_dir, majority_voting_folder, best_of_n_folder, weighted_majority_voting_folder):
    '''
    Compare the results of Majority Voting, Best-of-N, and Weighted Majority Voting
    and plot them on the same graph for each RM reward aggregation method (last, mean, min).
    
    Args:
        majority_voting_folder (str): Folder name of Majority Voting results.
        best_of_n_folder (str): Folder name of Best-of-N results.
        weighted_majority_voting_folder (str): Folder name of Weighted Majority Voting results.
    '''
    # Define the output directory
    # output_dir = os.path.join(results_dir, 'comparison', file_basename)
    # os.makedirs(output_dir, exist_ok=True)

    # Define RM reward aggregation methods
    aggregation_methods = ['last', 'mean', 'min']

    # Define file paths for each method
    majority_voting_path = os.path.join(results_dir, majority_voting_folder, file_basename)
    best_of_n_path = os.path.join(results_dir, best_of_n_folder, file_basename)
    weighted_majority_voting_path = os.path.join(results_dir, weighted_majority_voting_folder, file_basename)

    results = {'last': None, 'mean': None, 'min': None}

    for method in aggregation_methods:
        # Load metrics for Majority Voting
        majority_metrics_file = os.path.join(majority_voting_path, 'metrics.json')
        with open(majority_metrics_file, 'r', encoding='utf-8') as file:
            majority_metrics = json.load(file)

        # Load metrics for Best-of-N
        best_of_n_metrics_file = os.path.join(best_of_n_path, f'metrics_{method}.json')
        with open(best_of_n_metrics_file, 'r', encoding='utf-8') as file:
            best_of_n_metrics = json.load(file)

        # Load metrics for Weighted Majority Voting
        weighted_majority_voting_metrics_file = os.path.join(weighted_majority_voting_path, f'metrics_{method}.json')
        with open(weighted_majority_voting_metrics_file, 'r', encoding='utf-8') as file:
            weighted_majority_voting_metrics = json.load(file)

        # Extract data for plotting
        x = list(map(int, best_of_n_metrics.keys()))  # Sampling sizes (2^0, 2^1, ..., 2^8)
        majority_y = [majority_metrics[str(n)]['mean'] * 100 for n in x]  # Convert to percentages
        best_of_n_y = [best_of_n_metrics[str(n)]['mean'] * 100 for n in x]
        weighted_majority_voting_y = [weighted_majority_voting_metrics[str(n)]['mean'] * 100 for n in x]


        results[method] = (x, majority_y, best_of_n_y, weighted_majority_voting_y)

    return results

MODEL_NAMES = {'math_psa': 'Math-PSA', 
               'math_shepherd': 'Math-Shepherd', 
               'mmlu_noaugs_llama_lora': 'Llama3.1-8B Multi-Domain PRM', 'rlhflow_deepseek': 'RLHFLow-Deepseek', 
               'qwen2.5_math_7b_prm800k': 'Qwen-2.5-Math-PRM',
               'mmlu_math_noaugs_llama_lora': 'Llama3.1-8B Math-Subset PRM',
               'mmlu_small_noaugs_llama_lora': 'Llama3.1-8B Random-Subset PRM'}

Y_AXIS_LABEL_FONT_SIZE = 18
X_AXIS_LABEL_FONT_SIZE = 12
Y_AXIS_NUMBER_FONT_SIZE = 16
X_AXIS_NUMBER_FONT_SIZE = 16

BASELINE_COLOR = 'gray'
BASELINE_MARKER_SIZE = 10
MARKER_SIZE = 8
LEGEND_FONT_SIZE = 16

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_backup/results_by_category')
    parser.add_argument('--dataset', type=str, default='transformed_mmlupro')
    parser.add_argument('--models', type=list, nargs='*', default=['mmlu_noaugs_llama_lora', 'mmlu_math_noaugs_llama_lora', 'mmlu_small_noaugs_llama_lora'])
    parser.add_argument('--ignore', type=str, default='mmlu_overlap.json')
    parser.add_argument('--save_dir', type=str, default='section6')
    args = parser.parse_args()


    models_to_eval = {}
    if 'all' in args.models:
        folder_name_list = sorted(os.listdir(os.path.join(args.results_dir, 'comparison')))
        for folder_name in folder_name_list:
            model = folder_name.replace('cot_with_', '').replace('_rewards', '')
            if model.startswith('sciqqa') or model.startswith('v'):
                continue
            models_to_eval[model] = {}
        pass
    else:
        for model in args.models:
            models_to_eval[model] = {}

    # fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}
    # categories = ['all', 'all_except_math', 'health', 'computer science', 'economics', 'chemistry', 'business', 'other', 'physics', 'law', 'engineering', 'history', 'psychology', 'math', 'philosophy', 'biology']
    categories = {'math': 'Math', 'math_adjacent': 'Math-Adjacent Domains', 'non_math_adjacent': 'Non-Math-Adjacent Domains'}
    categories = {'math': 'Math', 'all_except_math': 'Non-Math Domains'}

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1, len(categories),
                           figsize=(12, 6)
                                )
    # cmap = plt.get_cmap('viridis', len(models_to_eval))
    # colors = [cmap(i) for i in range(len(models_to_eval))]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for c, category in enumerate(categories):
        
        lines = []
        fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}

        for model in models_to_eval:

            results = compare_results(file_basename = os.path.join('cot_with_{}_rewards'.format(model), category),
                                      results_dir=args.results_dir,
                                      majority_voting_folder='majority_voting_metrics',
                                      best_of_n_folder='best_of_n_metrics',
                                      weighted_majority_voting_folder='weighted_majority_voting_metrics'
                                      )
            models_to_eval[model][category] = {'data': results}
            flatten = {}
            for method in ['last', 'mean', 'min']:
                flatten[method] = []
                for i, item in enumerate(results[method]):
                    if i == 0:
                        continue
                    flatten[method] += item
                
                fig_y_max[method].append(max(flatten[method]))
                fig_y_min[method].append(min(flatten[method]))

        for method in ['min']:
            
            weighted_majority_voting_y_list = {}
            best_of_n_y_list = {}
            for model in models_to_eval:
                (x, majority_y, best_of_n_y, weighted_majority_voting_y) = models_to_eval[model][category]['data'][method]
                weighted_majority_voting_y_list[model] = weighted_majority_voting_y
                best_of_n_y_list[model] = best_of_n_y

            min_value = min(fig_y_min[method]) // 2 * 2 - 2
            max_value = max(fig_y_max[method]) // 2 * 2 + 2

            for m, model in enumerate(models_to_eval):
                line, = ax[c].plot(x, weighted_majority_voting_y_list[model], '-o', markersize=MARKER_SIZE, label='{} WMV'.format(MODEL_NAMES[model]), color=colors[m+4])
                lines.append(line)
            line, = ax[c].plot(x, majority_y, '-*', markersize=BASELINE_MARKER_SIZE, label='Majority Voting', color=BASELINE_COLOR)
            lines.append(line)

            ax[c].set_xscale('log', base=2)
            ax[c].set_ylim(min_value, max_value)
            ax[c].tick_params(axis='x', rotation=0, labelsize=X_AXIS_LABEL_FONT_SIZE)
            ax[c].tick_params(axis='y', labelsize=Y_AXIS_NUMBER_FONT_SIZE)
            ax[c].set_xticks(ticks=x, labels=[f'{n}' for n in x], fontsize=X_AXIS_NUMBER_FONT_SIZE)
            ax[c].set_xlabel('Number of sampled CoT solutions (log scale)', fontsize=Y_AXIS_LABEL_FONT_SIZE)
            ax[c].set_ylabel('Inference Accuracy (%)', fontsize=Y_AXIS_LABEL_FONT_SIZE)
            ax[c].set_title(f'{categories[category]}', fontsize=20)
            ax[c].grid(True)

            
    plt.tight_layout()

    custom_lines = [
        Line2D([0], [0], color=BASELINE_COLOR, lw=2, marker='*', markersize=14, linestyle='', label='Majority Voting'),
        Line2D([0], [0], color=BASELINE_COLOR, lw=2, marker='o', markersize=10, linestyle='', label='Weighted Majority Voting'),
        # Line2D([0], [0], color=BASELINE_COLOR, lw=2, marker='s', markersize=10, linestyle='', label='Best of N'),
    ]
    custom_lines2 = [
        Line2D([0], [0], color=colors[i+4], lw=6, linestyle='-', label='{} WMV'.format(MODEL_NAMES[model])) for i, model in enumerate(models_to_eval)
    ]
    # custom_lines.extend([
    #     Line2D([0], [0], color=colors[i], lw=6, linestyle='-', label='{} WMV'.format(MODEL_NAMES[model])) for i, model in enumerate(models_to_eval)
    # ])
    # custom_lines.extend([
    #     Line2D([0], [0], color=colors[i], lw=2, linestyle='--', label='{} BoN'.format(MODEL_NAMES[model])) for i, model in enumerate(models_to_eval)
    # ])

    fig.legend(handles=custom_lines, 
           labels=['Majority Voting', 'Weighted Majority Voting'],
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.025),  # 将图例放在 figure 的下方
           ncol=3,
           fontsize=15,
           frameon=False)
    plt.subplots_adjust(
    bottom=0.1,
    )

    fig.legend(handles=custom_lines2, 
           labels=[MODEL_NAMES[model] for model in models_to_eval],
           loc='lower center', 
           bbox_to_anchor=(0.5, -0.02),  # 将图例放在 figure 的下方
           ncol=3,
           fontsize=15,
           frameon=False)
    plt.subplots_adjust(
    bottom=0.2,
    )

    # fig.legend(handles=custom_lines, 
    #        labels=['Majority Voting', 'Weighted Majority Voting', 'Best of N'],
    #        loc='lower left', 
    #        bbox_to_anchor=(0, 0.04),  # 将图例放在 figure 的下方
    #        ncol=3,
    #        fontsize=LEGEND_FONT_SIZE,
    #        frameon=False)
    # plt.subplots_adjust(
    # bottom=0.2,
    # )

    # fig.legend(handles=custom_lines2, 
    #        labels=[MODEL_NAMES[model] for model in models_to_eval],
    #        loc='lower right', 
    #        bbox_to_anchor=(1, 0),  # 将图例放在 figure 的下方
    #        ncol=3,
    #        fontsize=LEGEND_FONT_SIZE,
    #        frameon=False)
    # plt.subplots_adjust(
    # bottom=0.2,
    # )

    # Save the plot
    os.makedirs(os.path.join(args.save_dir, 'subsection2'), exist_ok=True)
    plot_file_path = os.path.join(os.path.join(args.save_dir, 'subsection2'), f'prm_ours_{method}_agg.pdf')
    plt.savefig(plot_file_path)
    plt.close()