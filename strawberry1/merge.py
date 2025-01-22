import os, json
model = 'mmlu_small_noaugs_llama_lora'
dir = 'strawberry1/full_precision_results/transformed_mmlupro_reward_results/transformed_mmlupro_with_{}_reward/'.format(model)
files = os.listdir(dir)

data = []
for file in files:
    with open(os.path.join(dir, file), 'r', encoding='utf-8') as f:
        data.extend(json.load(f))

with open('strawberry1/full_precision_results/transformed_mmlupro_reward_results/transformed_mmlupro_with_{}_reward/cot_with_{}_rewards.json'.format(model, model), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)