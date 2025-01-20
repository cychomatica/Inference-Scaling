#!/bin/bash

DATASET="transformed_mmlupro"

# for MODEL in "v7_noaugs_llama_lora" "v7_onlyaugs_llama_lora" "mmlu_noaugs_llama_lora" "mmlu_onlyaugs_llama_lora" "sciqqa_onlyaugs_llama_lora"; do
for MODEL in "prm800k_llama_fulltune" "mmlu_noaugs_llama_lora"; do
    echo "running $MODEL"
    python calculate_metric_by_category.py --model=$MODEL
done

# tar -czvf /mnt/data/$DATASET\_by_category.tar.gz .