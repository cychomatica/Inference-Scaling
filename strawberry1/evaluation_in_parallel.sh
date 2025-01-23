#!/bin/bash
# set -e

declare -A CONFIGS=(
  ["./transformed_mmlupro"]="./full_precision_results/transformed_mmlupro_reward_results ./full_precision_figures"
)
MODELS=("mmlu_math_noaugs_llama_lora" "mmlu_small_noaugs_llama_lora")
BATCH_NUM=4

for DATASET in "${!CONFIGS[@]}"; do

  IFS=' ' read -r OUTPUT_DIR METRIC_FILE_DIR <<< "${CONFIGS[$DATASET]}"
  
  for MODEL in "${MODELS[@]}"; do

    PIDS=()
    mkdir -p $OUTPUT_DIR/log/$MODEL

    for BATCH_ID in {0..3}; do

      echo "Processing dataset $DATASET batch $(($BATCH_ID+1))/$BATCH_NUM with model: $MODEL"

      CUDA_VISIBLE_DEVICES=$BATCH_ID nohup \
      /opt/conda/envs/prm/bin/python -u ./search/get_rewards_reasoning_step_in_parallel.py \
      --example_file_path_dir "$DATASET" \
      --batch_id $BATCH_ID \
      --batch_num $BATCH_NUM \
      --test_prm "$MODEL" \
      --output_dir "$OUTPUT_DIR" \
      --metric_file_dir "$METRIC_FILE_DIR" \
      --do_not_calculate_metric \
      > $OUTPUT_DIR/log/$MODEL/$MODEL\_batch_$BATCH_ID.log 2>&1 &
      PIDS+=($!)

    done

    for PID_EVAL in "${PIDS[@]}"; do
        wait $PID_EVAL
        echo "Finished processing dataset $DATASET batch $(($BATCH_ID+1))/$BATCH_NUM with model: $MODEL"
    done

    # merge the batch outputs
    /opt/conda/envs/prm/bin/python \
    batch_outputs_merge.py \
    --model "$MODEL" \
    --batch_outputs_dir "$OUTPUT_DIR"
    echo "Finished merging batch outputs of model: $MODEL"

  done

done

echo "All tasks completed successfully."
