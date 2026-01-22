#!/bin/bash
MODEL_SIZE=${1:-"7"}
TASKS=${2:-"wikitext"}
SHOTS=${3:-"0"}
QUANT_MODE=${4:-"ant"}
QUANT_DTYPE=${5:-"int"}
GROUP_SIZE=${6:-"-1"}
QUANT_BIT_WIDTH=${7:-"w4a8k16v16"}
DESC=${8:-""}

MODEL=llama-${MODEL_SIZE}b-hf-transformers-4.29

OUTPUT_NAME=llama-${MODEL_SIZE}b
OUTPUT_DIR=output/output_llama

mkdir -p $OUTPUT_DIR

python -m run_evaluation --model_path $MODEL \
    --tasks $TASKS \
    --num_fewshot $SHOTS \
    --quant_bit_width $QUANT_BIT_WIDTH \
    --quant_mode $QUANT_MODE \
    --quant_dtype $QUANT_DTYPE \
    --q_group_size $GROUP_SIZE \
    | tee $OUTPUT_DIR/${OUTPUT_NAME}_${TASKS}_${QUANT_BIT_WIDTH}_${SHOTS}shots_${QUANT_MODE}_${QUANT_DTYPE}_g${GROUP_SIZE}_${DESC}_$(date +%m%d%H%M).log 2>&1
