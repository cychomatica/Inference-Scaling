import json
import os
import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import argparse



import os
import json
import matplotlib.pyplot as plt
import numpy as np


def compare_results(file_basename, results_dir, majority_voting_folder, best_of_n_folder, weighted_majority_voting_folder):
    """
    Compare the results of Majority Voting, Best-of-N, and Weighted Majority Voting
    and plot them on the same graph for each RM reward aggregation method (last, mean, min).
    
    Args:
        majority_voting_folder (str): Folder name of Majority Voting results.
        best_of_n_folder (str): Folder name of Best-of-N results.
        weighted_majority_voting_folder (str): Folder name of Weighted Majority Voting results.
    """
    # Define the output directory
    # script_path = os.path.abspath(__file__)
    # script_dir = os.path.dirname(script_path)
    # results_dir = "results_by_category"
    output_dir = os.path.join(results_dir, "comparison", file_basename)
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
        majority_metrics_file = os.path.join(majority_voting_path, "metrics.json")
        with open(majority_metrics_file, 'r', encoding='utf-8') as file:
            majority_metrics = json.load(file)

        # Load metrics for Best-of-N
        best_of_n_metrics_file = os.path.join(best_of_n_path, f"metrics_{method}.json")
        with open(best_of_n_metrics_file, 'r', encoding='utf-8') as file:
            best_of_n_metrics = json.load(file)

        # Load metrics for Weighted Majority Voting
        weighted_majority_voting_metrics_file = os.path.join(weighted_majority_voting_path, f"metrics_{method}.json")
        with open(weighted_majority_voting_metrics_file, 'r', encoding='utf-8') as file:
            weighted_majority_voting_metrics = json.load(file)

        # Extract data for plotting
        x = list(map(int, best_of_n_metrics.keys()))  # Sampling sizes (2^0, 2^1, ..., 2^8)
        majority_y = [majority_metrics[str(n)]["mean"] * 100 for n in x]  # Convert to percentages
        best_of_n_y = [best_of_n_metrics[str(n)]["mean"] * 100 for n in x]
        weighted_majority_voting_y = [weighted_majority_voting_metrics[str(n)]["mean"] * 100 for n in x]


        results[method] = (x, majority_y, best_of_n_y, weighted_majority_voting_y)

    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/mnt/data/results_by_category")
    parser.add_argument("--dataset", type=str, default="transformed_mmlupro")
    parser.add_argument("--models", type=list, nargs="*", default=['all'])
    parser.add_argument("--ignore", type=str, default='mmlu_overlap.json')
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

    categories = ["all", "all_except_math", "health", "computer science", "economics", "chemistry", "business", "other", "physics", "law", "engineering", "history", "psychology", "math", "philosophy", "biology"]

    fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}
    for category in categories:

        # fig_y_max, fig_y_min = {'last': [], 'mean': [], 'min': []}, {'last': [], 'mean': [], 'min': []}

        for model in models_to_eval:

            results = compare_results(file_basename = os.path.join('cot_with_{}_rewards'.format(model), category),
                                      results_dir=args.results_dir,
                                      majority_voting_folder="majority_voting_metrics",
                                      best_of_n_folder="best_of_n_metrics",
                                      weighted_majority_voting_folder="weighted_majority_voting_metrics"
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
            
            for model in models_to_eval:

                (x, majority_y, best_of_n_y, weighted_majority_voting_y) = models_to_eval[model][category]['data'][method]
                min_value = min(fig_y_min[method]) // 2 * 2 - 2
                max_value = max(fig_y_max[method]) // 2 * 2 + 2

                # Plot the results
                plt.figure(figsize=(8, 8))
                plt.plot(x, majority_y, '-o', label="Majority Voting", color="blue")
                plt.plot(x, best_of_n_y, '-o', label="Best-of-N", color="orange")
                plt.plot(x, weighted_majority_voting_y, '-o', label="Weighted Majority Voting", color="green")
                plt.xscale("log", base=2)
                # plt.xticks(x, labels=[f"$2^{{{int(np.log2(n))}}}$" for n in x])
                plt.xticks(x, labels=[f"{n}" for n in x])
                plt.xlabel("Number of sampled CoT solutions (log scale)")
                plt.ylim(min_value, max_value)
                plt.ylabel("Accuracy (%)")
                plt.title(f"Comparison of Voting Methods ({method.capitalize()} RM Reward Aggregation)")
                plt.legend()
                plt.grid(True)

                # Save the plot
                save_dir = os.path.join('new_figures', model, category)
                os.makedirs(save_dir, exist_ok=True)
                plot_file_path = os.path.join(save_dir, f"comparison_{method}.png")
                plt.savefig(plot_file_path)
                plt.close()