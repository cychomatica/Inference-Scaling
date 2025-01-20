import argparse, json, os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="strawberry1/full_precision_results")
    parser.add_argument("--dataset", type=str, default="transformed_mmlupro")
    parser.add_argument("--model", type=str, default="math_psa")
    parser.add_argument("--output_dir", type=str, default="results_by_category")
    args = parser.parse_args()

    file_dir = os.path.join(args.results_dir, "{}_reward_results".format(args.dataset), "{}_with_{}_reward".format(args.dataset, args.model))

    file_list = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f)) and os.path.join(file_dir, f).endswith(".json")]

    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        data_by_category = {}

        for i, obj in enumerate(data):

            if obj['metadata']['category'] not in data_by_category:
                data_by_category[obj['metadata']['category']] = [obj]
            else:
                data_by_category[obj['metadata']['category']].append(obj)

        data_by_category['all'] = data
    

    for category in data_by_category:
        output_dir = os.path.join(args.output_dir, args.dataset, args.model)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "category_{}.json".format(category)), 'w', encoding='utf-8') as json_file:
            json.dump(data_by_category[category], json_file, ensure_ascii=False, indent=4)