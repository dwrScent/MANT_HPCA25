#!/bin/bash

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT" || exit 1

# Default experiment configuration. Override through environment variables when needed.
DEFAULT_MODEL_NAMES=("qwen3-8b" "llama3-8b")
DEFAULT_MODEL_PATHS=(
    "${QWEN3_MODEL_PATH:-/cephfs/shared/model/Qwen3-8B}"
    "${LLAMA3_MODEL_PATH:-/cephfs/shared/model/llama-3-8b-hf}"
)
DEFAULT_METHODS=("olive" "ant" "mant" "meta-flint" "fp16")
DEFAULT_TASKS=(
    "hellaswag"
    "piqa"
    "winogrande"
    "arc_easy"
    "arc_challenge"
    "boolq"
)
DEFAULT_GROUP_SIZES=(64 32)

split_csv() {
    local csv="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$csv"
}

MODEL_NAMES=("${DEFAULT_MODEL_NAMES[@]}")
MODEL_PATHS=("${DEFAULT_MODEL_PATHS[@]}")
METHOD_LIST=("${DEFAULT_METHODS[@]}")
TASK_LIST=("${DEFAULT_TASKS[@]}")
GROUP_SIZE_LIST=("${DEFAULT_GROUP_SIZES[@]}")

if [ -n "${MODELS:-}" ]; then
    split_csv "$MODELS" MODEL_NAMES
fi
if [ -n "${MODEL_PATHS_CSV:-}" ]; then
    split_csv "$MODEL_PATHS_CSV" MODEL_PATHS
fi
if [ -n "${METHODS:-}" ]; then
    split_csv "$METHODS" METHOD_LIST
fi
if [ -n "${TASKS:-}" ]; then
    split_csv "$TASKS" TASK_LIST
fi
if [ -n "${GROUP_SIZES:-}" ]; then
    split_csv "$GROUP_SIZES" GROUP_SIZE_LIST
elif [ -n "${GROUP_SIZE:-}" ]; then
    GROUP_SIZE_LIST=("$GROUP_SIZE")
fi

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        PYTHON_BIN=python3
    fi
fi
SHOTS=${SHOTS:-0}
BATCH_SIZE=${BATCH_SIZE:-32}
BOOLQ_BATCH_SIZE=${BOOLQ_BATCH_SIZE:-8}
RECORD_BATCH_SIZE=${RECORD_BATCH_SIZE:-1}
QUANT_BIT_WIDTH=${QUANT_BIT_WIDTH:-"w4a4k16v16"}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-0}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-}
RUN_UPLOAD=${RUN_UPLOAD:-1}
UPLOAD_URL=${UPLOAD_URL:-"https://filebox.expectopatronum.cc/api/file?path="}
UPLOAD_TOKEN=${UPLOAD_TOKEN:-"lxy666"}

RUN_ID=${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-"$REPO_ROOT/output/auto_logs/$RUN_ID"}
RESULT_FILE=${RESULT_FILE:-"$REPO_ROOT/result.md"}
RESULTS_TSV="$LOG_DIR/results_${RUN_ID}.tsv"
SUMMARY_FILE="$LOG_DIR/summary_${RUN_ID}.txt"

mkdir -p "$LOG_DIR" "$(dirname "$RESULT_FILE")"

cleanup_gpu() {
    "$PYTHON_BIN" -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" >/dev/null 2>&1 || true
    sleep 2
}

sanitize_field() {
    local value="$1"
    value=${value//$'\t'/ }
    value=${value//$'\n'/ }
    printf "%s" "$value"
}

append_record() {
    local model_name="$1"
    local group_size="$2"
    local method="$3"
    local task="$4"
    local status="$5"
    local batch_size="$6"
    local log_file="$7"
    local metric="$8"
    local value="$9"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(sanitize_field "$model_name")" \
        "$(sanitize_field "$group_size")" \
        "$(sanitize_field "$method")" \
        "$(sanitize_field "$task")" \
        "$(sanitize_field "$status")" \
        "$(sanitize_field "$batch_size")" \
        "$(sanitize_field "$(basename "$log_file")")" \
        "$(sanitize_field "$metric")" \
        "$(sanitize_field "$value")" >>"$RESULTS_TSV"
}

parse_metrics() {
    local task="$1"
    local log_file="$2"

    "$PYTHON_BIN" - "$task" "$log_file" <<'PY'
import re
import sys
from pathlib import Path

task = sys.argv[1]
log_file = Path(sys.argv[2])
text = log_file.read_text(encoding="utf-8", errors="ignore")
rows = []
current_task = None
header = None

def norm(name):
    return re.sub(r"[^a-z0-9]+", "", name.lower())

for line in text.splitlines():
    if "|" not in line:
        continue
    stripped = line.strip()
    if not stripped.startswith("|"):
        continue
    parts = [p.strip() for p in stripped.split("|")[1:-1]]
    if not parts or all(not p for p in parts):
        continue
    if all(set(p) <= {"-", ":"} for p in parts if p):
        continue

    normalized = [norm(p) for p in parts]
    if "metric" in normalized and "value" in normalized:
        header = normalized
        continue
    if header is None:
        continue

    task_idx = next((i for i, p in enumerate(header) if p in {"task", "tasks"}), None)
    metric_idx = header.index("metric")
    value_idx = header.index("value")
    if len(parts) <= max(metric_idx, value_idx):
        continue

    task_name = parts[task_idx] if task_idx is not None and task_idx < len(parts) else ""
    if task_name:
        current_task = task_name
    metric_name = parts[metric_idx]
    value = parts[value_idx]
    if current_task == task and metric_name and value:
        rows.append((metric_name, value))

if not rows:
    m = re.findall(r"(?m)^Task:\s*([^,]+),\s*PPL:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
    for name, value in m:
        if name.strip() == task:
            rows.append(("ppl", value))
    if not rows:
        m = re.findall(r"(?m)^([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if m:
            rows.append(("ppl", m[-1]))

seen = set()
for metric_name, value in rows:
    if metric_name in seen:
        continue
    seen.add(metric_name)
    print(f"{metric_name}\t{value}")
PY
}

classify_failure() {
    local log_file="$1"

    if grep -qi "out of memory\|cuda error\|killed\|interrupt\|KeyboardInterrupt\|RuntimeError\|trust_remote_code=True\|empty range for randrange\|FileNotFoundError\|ImportError\|ModuleNotFoundError" "$log_file"; then
        echo "SKIP"
    else
        echo "FAIL"
    fi
}

method_config() {
    local method="$1"
    local group_size="$2"

    METHOD_QUANT_MODE="$method"
    METHOD_QUANT_DTYPE="int"
    METHOD_QUANT_BIT_WIDTH="$QUANT_BIT_WIDTH"
    METHOD_GROUP_SIZE="$group_size"

    case "$method" in
        fp16)
            METHOD_QUANT_MODE="int"
            METHOD_QUANT_DTYPE="int"
            METHOD_QUANT_BIT_WIDTH="${FP16_QUANT_BIT_WIDTH:-w16a16k16v16}"
            METHOD_GROUP_SIZE="${FP16_GROUP_SIZE:--1}"
            ;;
        olive)
            METHOD_QUANT_MODE="olive"
            METHOD_QUANT_DTYPE="${OLIVE_QUANT_DTYPE:-int-flint}"
            METHOD_QUANT_BIT_WIDTH="${OLIVE_QUANT_BIT_WIDTH:-$QUANT_BIT_WIDTH}"
            METHOD_GROUP_SIZE="${OLIVE_GROUP_SIZE:-$group_size}"
            ;;
        ant)
            METHOD_QUANT_MODE="ant"
            METHOD_QUANT_DTYPE="${ANT_QUANT_DTYPE:-int-flint-pot-float}"
            METHOD_QUANT_BIT_WIDTH="${ANT_QUANT_BIT_WIDTH:-$QUANT_BIT_WIDTH}"
            METHOD_GROUP_SIZE="${ANT_GROUP_SIZE:-$group_size}"
            ;;
        mant)
            METHOD_QUANT_MODE="mant"
            METHOD_QUANT_DTYPE="${MANT_QUANT_DTYPE:-int}"
            METHOD_QUANT_BIT_WIDTH="${MANT_QUANT_BIT_WIDTH:-$QUANT_BIT_WIDTH}"
            METHOD_GROUP_SIZE="${MANT_GROUP_SIZE:-$group_size}"
            ;;
        meta-flint|meta_flint|metaflint)
            METHOD_QUANT_MODE="meta-flint"
            METHOD_QUANT_DTYPE="${META_FLINT_QUANT_DTYPE:-int}"
            METHOD_QUANT_BIT_WIDTH="${META_FLINT_QUANT_BIT_WIDTH:-$QUANT_BIT_WIDTH}"
            METHOD_GROUP_SIZE="${META_FLINT_GROUP_SIZE:-$group_size}"
            ;;
        *)
            echo "Unknown method: $method" >&2
            return 1
            ;;
    esac
}

render_result_file() {
    "$PYTHON_BIN" - "$RESULTS_TSV" "$RESULT_FILE" "$RUN_ID" "$QUANT_BIT_WIDTH" "${GROUP_SIZE_LIST[*]}" "$SHOTS" "$BATCH_SIZE" "$BOOLQ_BATCH_SIZE" "$RECORD_BATCH_SIZE" "$LOG_DIR" "${METHOD_LIST[*]}" "${MODEL_NAMES[*]}" "${TASK_LIST[*]}" <<'PY'
import csv
import sys
from datetime import datetime
from pathlib import Path

results_tsv = Path(sys.argv[1])
result_file = Path(sys.argv[2])
run_id = sys.argv[3]
quant_bit_width = sys.argv[4]
group_sizes = sys.argv[5].split()
shots = sys.argv[6]
batch_size = sys.argv[7]
boolq_batch_size = sys.argv[8]
record_batch_size = sys.argv[9]
log_dir = sys.argv[10]
methods = sys.argv[11].split()
models = sys.argv[12].split()
tasks = sys.argv[13].split()

primary_metric = {
    "hellaswag": "acc_norm",
    "piqa": "acc_norm",
    "winogrande": "acc",
    "arc_easy": "acc_norm",
    "arc_challenge": "acc_norm",
    "boolq": "acc",
}

metrics = {}
statuses = {}
logs = {}

if results_tsv.exists():
    with results_tsv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            group_size = row.get("group_size", "")
            key = (group_size, row["model"], row["method"], row["task"])
            statuses[key] = row["status"]
            logs[key] = row["log"]
            metrics[(group_size, row["model"], row["method"], row["task"], row["metric"])] = row["value"]

def as_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.upper() == "N/A":
            return None
        return float(text)
    except ValueError:
        return None

def format_value(value, percent=True, decimals=2):
    number = as_float(value)
    if number is None:
        return "" if value in (None, "", "N/A") else str(value)
    if percent and abs(number) <= 1.0:
        number *= 100.0
    return f"{number:.{decimals}f}"

def metric_value(group_size, model, method, task, metric):
    value = metrics.get((group_size, model, method, task, metric))
    if value is not None:
        return value
    status = statuses.get((group_size, model, method, task))
    if status and status != "OK":
        return status
    return ""

def first_metric_value(group_size, model, method, task):
    preferred = primary_metric.get(task, "acc")
    for metric in [preferred, "acc_norm", "acc", "f1", "em", "ppl"]:
        value = metric_value(group_size, model, method, task, metric)
        if value != "":
            return metric, value
    return preferred, ""

def average(values, percent=True):
    nums = []
    for value in values:
        number = as_float(value)
        if number is None:
            continue
        if percent and abs(number) <= 1.0:
            number *= 100.0
        nums.append(number)
    if not nums:
        return ""
    return f"{sum(nums) / len(nums):.2f}"

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)

def downstream_table(group_size, model):
    rows = []
    for method in methods:
        raw_values = []
        cells = []
        for task in tasks:
            metric, value = first_metric_value(group_size, model, method, task)
            raw_values.append(value if metric != "ppl" else "")
            cell = format_value(value, percent=(metric != "ppl"), decimals=2 if metric != "ppl" else 3)
            if cell and cell not in {"FAIL", "SKIP"} and metric not in {"", primary_metric.get(task, "acc")}:
                cell = f"{cell} {metric}"
            cells.append(cell)
        rows.append([method, *cells, average(raw_values)])
    return md_table(["Method", *tasks, "avg"], rows)

def metric_detail_table(group_size, model, metric_name):
    rows = []
    has_any = False
    for method in methods:
        raw_values = [metric_value(group_size, model, method, task, metric_name) for task in tasks]
        if any(value not in {"", "FAIL", "SKIP"} for value in raw_values):
            has_any = True
        rows.append([method, *[format_value(value) for value in raw_values], average(raw_values)])
    if not has_any:
        return ""
    return md_table(["Method", *tasks, "avg"], rows)

def failed_rows():
    rows = []
    for group_size in group_sizes:
        for model in models:
            for method in methods:
                for task in tasks:
                    status = statuses.get((group_size, model, method, task))
                    if status and status != "OK":
                        rows.append([group_size, model, method, task, status, logs.get((group_size, model, method, task), "")])
    return rows

lines = [
    "# MANT Downstream Evaluation Result",
    "",
    f"- Run id: {run_id}",
    f"- Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"- Methods: {', '.join(methods)}",
    f"- Default quant bit width: {quant_bit_width}",
    f"- Group sizes: {', '.join(group_sizes)}",
    f"- Few-shot: {shots}",
    f"- Default batch size: {batch_size}",
    f"- BoolQ batch size: {boolq_batch_size}",
    f"- ReCoRD batch size: {record_batch_size}",
    f"- Models: {', '.join(models)}",
    f"- Tasks: {', '.join(tasks)}",
    f"- Log dir: {log_dir}",
    "",
]

for group_size in group_sizes:
    lines.extend([f"## Group Size {group_size}", ""])
    for model in models:
        lines.extend([f"### {model}", "", "#### Primary Metric", "", downstream_table(group_size, model), ""])
        acc_norm = metric_detail_table(group_size, model, "acc_norm")
        if acc_norm:
            lines.extend(["#### Accuracy Norm", "", acc_norm, ""])

failures = failed_rows()
if failures:
    lines.extend(["## Failed or Skipped Tasks", "", md_table(["Group", "Model", "Method", "Task", "Status", "Log"], failures), ""])

result_file.write_text("\n".join(lines), encoding="utf-8")
PY
}

init_result_file() {
    printf "model\tgroup_size\tmethod\ttask\tstatus\tbatch\tlog\tmetric\tvalue\n" >"$RESULTS_TSV"
    render_result_file
}

run_one_task() {
    local model_name="$1"
    local model_path="$2"
    local group_size="$3"
    local method="$4"
    local task="$5"
    local safe_model_name="${model_name//[^A-Za-z0-9_.-]/_}"
    local safe_group_size="${group_size//[^A-Za-z0-9_.-]/_}"
    local safe_method="${method//[^A-Za-z0-9_.-]/_}"
    local safe_task="${task//[^A-Za-z0-9_.-]/_}"
    local log_file="$LOG_DIR/${RUN_ID}_${safe_model_name}_g${safe_group_size}_${safe_method}_${safe_task}.log"
    local status="OK"
    local result="metric=N/A"
    local current_batch_size="$BATCH_SIZE"
    local exit_code=1

    if [ "$task" = "boolq" ]; then
        current_batch_size="$BOOLQ_BATCH_SIZE"
    elif [ "$task" = "record" ]; then
        current_batch_size="$RECORD_BATCH_SIZE"
    fi

    if ! method_config "$method" "$group_size"; then
        status="FAIL"
        result="unknown_method"
        append_record "$model_name" "$group_size" "$method" "$task" "$status" "$current_batch_size" "$log_file" "error" "$result"
        render_result_file
        return 0
    fi

    echo
    echo "========== Running: group_size=$group_size model=$model_name method=$method task=$task =========="
    echo "log: $log_file"
    echo "config: quant_mode=$METHOD_QUANT_MODE quant_dtype=$METHOD_QUANT_DTYPE quant_bit_width=$METHOD_QUANT_BIT_WIDTH group_size=$METHOD_GROUP_SIZE"

    if [ ! -d "$model_path" ]; then
        status="SKIP"
        result="model_path_not_found: $model_path"
        printf "%-8s %-12s %-12s %-16s %-8s %s\n" "$group_size" "$model_name" "$method" "$task" "$status" "$result" | tee -a "$SUMMARY_FILE"
        append_record "$model_name" "$group_size" "$method" "$task" "$status" "$current_batch_size" "$log_file" "error" "$result"
        render_result_file
        return 0
    fi

    local cmd=(
        "$PYTHON_BIN" -m run_evaluation
        --model_path "$model_path"
        --tasks "$task"
        --batch_size "$current_batch_size"
        --num_fewshot "$SHOTS"
        --quant_bit_width "$METHOD_QUANT_BIT_WIDTH"
        --quant_mode "$METHOD_QUANT_MODE"
        --quant_dtype "$METHOD_QUANT_DTYPE"
        --q_group_size "$METHOD_GROUP_SIZE"
    )
    if [ -n "$LIMIT_SAMPLES" ]; then
        cmd+=(--limit_samples "$LIMIT_SAMPLES")
    fi

    if [ "$TIMEOUT_SECONDS" -gt 0 ]; then
        timeout "$TIMEOUT_SECONDS" "${cmd[@]}" >"$log_file" 2>&1
    else
        "${cmd[@]}" >"$log_file" 2>&1
    fi
    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        local parsed_any=0
        while IFS=$'\t' read -r metric value; do
            if [ -z "${metric:-}" ]; then
                continue
            fi
            parsed_any=1
            append_record "$model_name" "$group_size" "$method" "$task" "$status" "$current_batch_size" "$log_file" "$metric" "$value"
            if [ "$result" = "metric=N/A" ]; then
                result="$metric=$value"
            else
                result="$result $metric=$value"
            fi
        done < <(parse_metrics "$task" "$log_file")

        if [ "$parsed_any" -eq 0 ]; then
            result="metric=N/A"
            append_record "$model_name" "$group_size" "$method" "$task" "$status" "$current_batch_size" "$log_file" "metric" "N/A"
        fi
    else
        status=$(classify_failure "$log_file")
        result="see $(basename "$log_file")"
        append_record "$model_name" "$group_size" "$method" "$task" "$status" "$current_batch_size" "$log_file" "error" "$result"
    fi

    cleanup_gpu
    render_result_file

    printf "%-8s %-12s %-12s %-16s %-8s %s (bs=%s)\n" "$group_size" "$model_name" "$method" "$task" "$status" "$result" "$current_batch_size" | tee -a "$SUMMARY_FILE"
}

upload_result() {
    if [ "$RUN_UPLOAD" != "1" ]; then
        return 0
    fi

    echo
    echo "========== Uploading $(basename "$RESULT_FILE") =========="
    curl -fS -X PUT "$UPLOAD_URL" \
        -H "Authorization: Bearer $UPLOAD_TOKEN" \
        -H "X-Filename: $(basename "$RESULT_FILE")" \
        --data-binary @"$RESULT_FILE"
}

if [ "${#MODEL_NAMES[@]}" -ne "${#MODEL_PATHS[@]}" ]; then
    echo "MODEL_NAMES and MODEL_PATHS length mismatch" >&2
    exit 1
fi

init_result_file

{
    echo "Run id: $RUN_ID"
    echo "Result file: $RESULT_FILE"
    echo "Log dir: $LOG_DIR"
    echo "Default quant bit width: $QUANT_BIT_WIDTH"
    echo "Group sizes: ${GROUP_SIZE_LIST[*]}"
    echo "Tasks: ${TASK_LIST[*]}"
    echo
    printf "%-8s %-12s %-12s %-16s %-8s %s\n" "Group" "Model" "Method" "Task" "Status" "Result"
    printf "%-8s %-12s %-12s %-16s %-8s %s\n" "-----" "-----" "------" "----" "------" "------"
} | tee "$SUMMARY_FILE"

for group_size in "${GROUP_SIZE_LIST[@]}"; do
    for model_index in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$model_index]}"
        model_path="${MODEL_PATHS[$model_index]}"
        for method in "${METHOD_LIST[@]}"; do
            for task in "${TASK_LIST[@]}"; do
                run_one_task "$model_name" "$model_path" "$group_size" "$method" "$task"
            done

            render_result_file
            if ! upload_result; then
                echo "Upload failed after group_size=$group_size model=$model_name method=$method; result.md is still available at $RESULT_FILE" >&2
            fi
        done
    done
done

echo
echo "========== Summary =========="
cat "$SUMMARY_FILE"
echo
echo "Result file: $RESULT_FILE"
