#!/bin/bash

DATASET="transformed_mmlupro"

for MODEL in "deepseek_8b_prm" "math_psa" "prm800k_llama_fulltune" "prm800k_qwen_fulltune" "sciqqa_noaugs_llama_lora" "sciqqa_noaugs_qwen_lora" "v4_llama_lora" "v4_qwen_lora" "v5_llama_lora" "v5_qwen_lora"; do
    python results_split_by_category.py --model=$MODEL
    echo "Model $MODEL done."
done

tar -czvf /mnt/data/$DATASET\_by_category.tar.gz .
