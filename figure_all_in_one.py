import argparse
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FormatStrFormatter

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
    # script_path = os.path.abspath(__file__)
    # script_dir = os.path.dirname(script_path)
    # results_dir = 'results_by_category'
    output_dir = os.path.join(results_dir, 'comparison', file_basename)
    os.makedirs(output_dir, exist_ok=True)

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
               'qwen2.5_math_7b_prm800k': 'Qwen2.5-Math-7B\nPRM800K',
               'rlhflow_deepseek': 'RLHFlow-Llama3.1-8B\nDeepseek',
               'prm800k_llama_fulltune': 'Llama-3.1-8B-Instruct\nFulltuning on PRM800K',
               'mmlu_noaugs_llamabase_lora': 'Llama-3.1-8B-Instruct \n+ LoRA on MMLU-Pro-Train',
               'mmlu_noaugs_llama_lora': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ LoRA on MMLU-Pro-Train',
               'mmlu_augs_llama_lora': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ LoRA on MMLU-Pro-Train \n& MMLU-Pro Counterfactual Augmentations',
               'mmlu_onlyaugs_llama_lora': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ MMLU-Pro Counterfactual Augmentations',
               'mmlu_small_noaugs_llama_lora': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ LoRA on MMLU-Pro-Train Random Subset',
               'mmlu_math_noaugs_llama_lora': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ LoRA on MMLU-Pro-Train Math Subset',
               'mmlu_noaugs_llama_fulltune': 'Llama-3.1-8B-Instruct \n+ Fulltuning on PRM800K \n+ Fulltuning on MMLU-Pro-Train',
               'prm800k_qwen_fulltune': 'Qwen2.5-Math-7B-Instruct \n+ Fulltuning on PRM800K',
               'mmlu_noaugs_qwen_lora': 'Qwen2.5-Math-7B-Instruct \n+ Fulltuning on PRM800K \n+ LoRA on MMLU-Pro-Train',
               }

Y_AXIS_LABEL_FONT_SIZE = 20
X_AXIS_LABEL_FONT_SIZE = 12
Y_AXIS_NUMBER_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results_backup/results_by_category')
    parser.add_argument('--dataset', type=str, default='transformed_mmlupro')
    parser.add_argument('--models', type=list, nargs='*', default=['all'])
    parser.add_argument('--ignore', type=str, default='mmlu_overlap.json')
    parser.add_argument('--save_dir', type=str, default='section1')
    args = parser.parse_args()


    models_to_eval = {}
    for model in ['math_psa', 'math_shepherd', 'qwen2.5_math_7b_prm800k', 'rlhflow_deepseek', 'prm800k_llama_fulltune', 'mmlu_noaugs_llama_lora']:
        models_to_eval[model] = {}

    categories = ['all', 'all_except_math', 'health', 'computer science', 'economics', 'chemistry', 'business', 'other', 'physics', 'law', 'engineering', 'history', 'psychology', 'math', 'philosophy', 'biology']

    fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}
    for category in categories:

        # fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}

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

            # models_to_eval[model][category]['max_value'] = max(flatten)
            # models_to_eval[model][category]['min_value'] = min(flatten)
            # fig_y_max.append(models_to_eval[model][category]['max_value'])
            # fig_y_min.append(models_to_eval[model][category]['min_value'])
            
    for category in categories:

        for method in ['last', 'mean', 'min']:
            
            weighted_majority_voting_y_list = {}
            for model in models_to_eval:
                (x, majority_y, best_of_n_y, weighted_majority_voting_y) = models_to_eval[model][category]['data'][method]
                weighted_majority_voting_y_list[model] = weighted_majority_voting_y

            min_value = 40 # min(fig_y_min[method]) // 2 * 2 - 5
            max_value = max(fig_y_max[method]) // 2 * 2 + 5

            # Plot the results
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.rcParams["font.family"] = "Times New Roman"

            classes = []
            values = []
            for model in models_to_eval:
                classes.append(MODEL_NAMES[model])
                values.append(weighted_majority_voting_y_list[model][-1])

            cmap = plt.get_cmap('viridis', len(models_to_eval))
            # colors = [cmap(i) for i in range(len(models_to_eval))]
            # colors = [cmap(0) for i in range(5)] + [cmap(i) for i in range(5, len(models_to_eval))]
            colors = [cmap(0) for i in range(5)] + [cmap(i) for i in range(5, len(models_to_eval))]

            bars = ax.bar(classes[:-1], values[:-1], color=colors[:-1], width=0.6)
            ax.axhline(y=majority_y[-1], color='#FF5733', linestyle='--', linewidth=2, label='Majority Voting: {:.2f}'.format(majority_y[-1]))

            # ax.text(x=-0.6,
            #         y=majority_y[-1] - 1, 
            #         s='{:.2f}'.format(majority_y[-1]),
            #         color='#FF5733',
            #         ha='right',
            #         va='bottom',
            #         fontsize=AXIS_NUMBER_FONT_SIZE)
            # plt.text(10, baseline_y + 0.1, 'Baseline (y=0)', color='red', fontsize=10, ha='right')


            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', 
                        ha='center', va='bottom', fontsize=Y_AXIS_NUMBER_FONT_SIZE)
            
            idx_improvement = len(classes) - 2
            ax.bar(idx_improvement, values[-1] - values[idx_improvement], bottom=values[idx_improvement], color=colors[-1], label='LoRA on MMLU-Pro-Train',  width=0.6)
            ax.text(idx_improvement, values[-1] + 0.01,
                    f'{values[-1]:.2f}', 
                    ha='center', va='bottom', fontsize=Y_AXIS_NUMBER_FONT_SIZE)
            
            ax.set_ylim(min_value, max_value)
            ax.tick_params(axis='y', labelsize=Y_AXIS_NUMBER_FONT_SIZE)
            ax.set_ylabel('Accuracy (%)', fontsize=Y_AXIS_LABEL_FONT_SIZE)
            ax.tick_params(axis='x', rotation=0, labelsize=X_AXIS_LABEL_FONT_SIZE)
            # ax.set_title('{} Domain Weighted Majority Voting (N=128, {} RM Reward Aggregation)'.format(category.capitalize(), method.capitalize()), fontsize=TITILE_FONT_SIZE)
            ax.legend(loc='upper left', fontsize=LEGEND_FONT_SIZE)
            plt.tight_layout()

            # Save the plot
            save_dir = os.path.join(args.save_dir, category)
            os.makedirs(save_dir, exist_ok=True)
            plot_file_path = os.path.join(save_dir, '{}_wmv_{}_agg.pdf'.format(category, method))
            plt.savefig(plot_file_path)
            plt.close() 