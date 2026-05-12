#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_MODEL_PATH="/cephfs/shared/model/llama-3-8b-hf"

TARGET_METRIC=${1:-""}
MODEL_PATH=${MODEL_PATH:-"${DEFAULT_MODEL_PATH}"}
TASKS=${TASKS:-"wikitext"}
SHOTS=${SHOTS:-"0"}
# METHODS=${METHODS:-"olive,ant,mant"}
METHODS=${METHODS:-"olive"}
SAMPLE=${SAMPLE:-"64"}
SOTA_SAMPLE=${SOTA_SAMPLE:-"${SAMPLE}"}
INITIAL_MODEL_LAYER_CONFIG=${INITIAL_MODEL_LAYER_CONFIG:-"example"}
BATCH_SIZE=${BATCH_SIZE:-"32"}
LAYER_A_BITS=${LAYER_A_BITS:-"follow"}
MAX_CANDIDATES_PER_STEP=${MAX_CANDIDATES_PER_STEP:-"8"}
MAX_EVALS=${MAX_EVALS:-"0"}
DEFAULT_SOTA_PYTHON="/root/miniconda3/envs/mxq/bin/python"
if [[ -x "${DEFAULT_SOTA_PYTHON}" ]]; then
    SOTA_PYTHON=${SOTA_PYTHON:-"${DEFAULT_SOTA_PYTHON}"}
else
    SOTA_PYTHON=${SOTA_PYTHON:-"python"}
fi

cd "${REPO_ROOT}"

CMD=(python scripts/search_precision_config.py
    --model_path "${MODEL_PATH}" \
    --tasks "${TASKS}" \
    --methods "${METHODS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_fewshot "${SHOTS}" \
    --limit_samples "${SAMPLE}" \
    --sota_limit_samples "${SOTA_SAMPLE}" \
    --initial_model_layer_config "${INITIAL_MODEL_LAYER_CONFIG}" \
    --layer_a_bits "${LAYER_A_BITS}" \
    --sota_python "${SOTA_PYTHON}" \
    --max_candidates_per_step "${MAX_CANDIDATES_PER_STEP}" \
    --max_evals "${MAX_EVALS}")

if [[ -n "${TARGET_METRIC}" ]]; then
    CMD+=(--target_metric "${TARGET_METRIC}")
fi

"${CMD[@]}"
