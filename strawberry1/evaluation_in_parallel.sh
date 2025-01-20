#!/bin/bash
# set -e

# Define an associative array mapping datasets to their corresponding output directories and metric file directories
declare -A CONFIGS=(
  ["./transformed_mmlupro"]="./full_precision_results/transformed_mmlupro_reward_results ./full_precision_figures"
)
MODEL=("mmlu_augs_llama_lora")
BATCH_ID=0
BATCH_NUM=4

# Loop over each dataset configuration
for DATASET in "${!CONFIGS[@]}"; do
  # Split the configuration value into output directory and metric file directory
  IFS=' ' read -r OUTPUT_DIR METRIC_FILE_DIR <<< "${CONFIGS[$DATASET]}"
  
  # Loop over each model
  for MODEL in "${MODELS[@]}"; do
    echo "Processing dataset $DATASET batch $BATCH_ID/$BATCH_NUM with model: $MODEL"
    
    # Run the Python script with current dataset, model, and output configuration
    CUDA_VISIBLE_DEVICES=$BATCH_ID \
    python ./search/get_rewards_reasoning_step_in_parallel.py \
      --example_file_path_dir "$DATASET" \
      --batch_id $BATCH_ID \
      --batch_num $BATCH_NUM \
      --test_prm "$MODEL" \
      --output_dir "$OUTPUT_DIR" \
      --metric_file_dir "$METRIC_FILE_DIR" \
      --do_not_calculate_metric \
    
    echo "Finished processing dataset $DATASET batch $BATCH_ID/$BATCH_NUM with model: $MODEL"
  done
done

echo "All tasks completed successfully."
