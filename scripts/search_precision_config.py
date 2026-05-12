#!/usr/bin/env python3
"""Search mixed-precision bit patterns against an nvesm2 target metric."""

from __future__ import annotations

import argparse
import ast
import errno
import json
import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


PPL_TASKS = {"wikitext", "c4", "ptb"}
DEFAULT_METHOD_DTYPES = {
    "ant": "int-flint-pot-float",
    "olive": "int-flint",
    "mant": "int",
}
COLOR_ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def color(text: str, code: str) -> str:
    if not COLOR_ENABLED:
        return text
    return f"\033[{code}m{text}\033[0m"


def section(title: str, detail: str = "", code: str = "1;36") -> None:
    line = f"=== {title} ==="
    if detail:
        line = f"{line} {detail}"
    print(f"\n{color(line, code)}", flush=True)


def important(label: str, detail: str, code: str = "1;32") -> None:
    print(f"\n{color(f'=== {label} ===', code)} {detail}\n", flush=True)


def bits_display(bits: List[int], high_bit: int = 8) -> str:
    return compact_bits_expr(bits) or f"mixed high_bits={bits.count(high_bit)} total={len(bits)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find ant/olive/mant per-linear bit patterns with metric close to nvesm2."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", default="wikitext")
    parser.add_argument("--metric", choices=["auto", "ppl", "score"], default="auto")
    parser.add_argument("--target_metric", type=float, default=None)
    parser.add_argument("--methods", default="olive,ant,mant")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--num_tensors", type=int, default=None)
    parser.add_argument("--tensors_per_layer", type=int, default=None)
    parser.add_argument(
        "--initial_model_layer_config",
        "--initial_layer_bit_config",
        dest="initial_layer_bit_config",
        choices=["example", "uniform"],
        default="example",
        help="Use accel_model_configs_example.py as the initial bit pattern, or start uniformly.",
    )
    parser.add_argument("--initial_bit", type=int, default=4)
    parser.add_argument("--high_bit", type=int, default=8)
    parser.add_argument("--abs_tolerance", type=float, default=0.02)
    parser.add_argument("--rel_tolerance", type=float, default=0.001)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--max_evals", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--max_candidates_per_step", type=int, default=0, help="0 means test all candidates.")
    parser.add_argument("--layer_a_bits", choices=["global", "follow"], default="follow")
    parser.add_argument("--a_bit", type=int, default=4)
    parser.add_argument("--k_bit", type=int, default=16)
    parser.add_argument("--v_bit", type=int, default=16)
    parser.add_argument("--method_group_size", type=int, default=64)
    parser.add_argument("--mant_group_size", type=int, default=64)
    parser.add_argument("--ant_dtype", default=DEFAULT_METHOD_DTYPES["ant"])
    parser.add_argument("--olive_dtype", default=DEFAULT_METHOD_DTYPES["olive"])
    parser.add_argument("--mant_dtype", default=DEFAULT_METHOD_DTYPES["mant"])
    parser.add_argument("--sota_repo", default="../mxfp_quant/pseudo_quantization")
    parser.add_argument(
        "--sota_python",
        default=os.environ.get("SOTA_PYTHON", sys.executable),
        help="Python executable used to evaluate the nvesm2 SOTA target.",
    )
    parser.add_argument(
        "--sota_limit_samples",
        type=int,
        default=None,
        help="Limit samples for the nvesm2 SOTA target. Defaults to full SOTA evaluation.",
    )
    parser.add_argument("--sota_w_bit", type=int, default=4)
    parser.add_argument("--sota_a_bit", type=int, default=4)
    parser.add_argument("--sota_w_mode", default="nvesm2")
    parser.add_argument("--sota_a_mode", default="nvesm2")
    parser.add_argument("--sota_group_size", type=int, default=16)
    parser.add_argument("--output_dir", default="output/precision_search")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit_samples must be a positive integer")
    if args.sota_limit_samples is not None and args.sota_limit_samples <= 0:
        raise ValueError("--sota_limit_samples must be a positive integer")
    return args


def metric_kind(args: argparse.Namespace) -> str:
    if args.metric != "auto":
        return args.metric
    return "ppl" if args.tasks in PPL_TASKS else "score"


def lower_is_better(kind: str) -> bool:
    return kind == "ppl"


def parse_metric_from_log(text: str, kind: str, task_name: str) -> float:
    if kind == "ppl":
        matches = re.findall(r"(?m)^([0-9]+(?:\.[0-9]+)?)\s*$", text)
        if not matches:
            raise ValueError("Could not parse PPL from log")
        return float(matches[-1])

    preferred = {
        "hellaswag": "acc_norm",
        "piqa": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "boolq": "acc",
    }
    task = task_name.split(",")[0]
    metric = preferred.get(task)
    rows = parse_lm_eval_rows(text)
    if metric is not None:
        for row_task, row_metric, value in rows:
            if row_task == task and row_metric == metric:
                return float(value)
    if rows:
        return float(rows[0][2])
    raise ValueError("Could not parse score from lm_eval table")


def parse_lm_eval_rows(text: str) -> List[Tuple[str, str, str]]:
    rows = []
    current_task = None
    for line in text.splitlines():
        if "|" not in line or "Metric" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")[1:-1]]
        if len(parts) < 7:
            continue
        if parts[0]:
            current_task = parts[0]
        if current_task and parts[4] and re.match(r"^-?[0-9]+(?:\.[0-9]+)?$", parts[6]):
            rows.append((current_task, parts[4], parts[6]))
    return rows


def run_command(cmd: List[str], cwd: Path, log_file: Path, dry_run: bool) -> float:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    section("RUN", f"cwd={cwd}", "1;34")
    print(color(printable, "36"), flush=True)
    print(color(f"log: {log_file}", "2"), flush=True)
    if dry_run:
        log_file.write_text(printable + "\n", encoding="utf-8")
        return 0.0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    with log_file.open("w", encoding="utf-8") as f:
        proc = stream_process(cmd, cwd, env, f)
    if proc.returncode != 0:
        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. Log: {log_file}\n{tail}")
    return proc.returncode


def write_stdout_bytes(data: bytes) -> None:
    try:
        os.write(sys.stdout.fileno(), data)
    except (AttributeError, OSError, ValueError):
        print(data.decode("utf-8", errors="replace"), end="", flush=True)


def stream_process(cmd: List[str], cwd: Path, env: Dict[str, str], log_file) -> subprocess.Popen:
    if sys.stdout.isatty():
        return stream_process_pty(cmd, cwd, env, log_file)
    return stream_process_pipe(cmd, cwd, env, log_file)


def stream_process_pipe(cmd: List[str], cwd: Path, env: Dict[str, str], log_file) -> subprocess.Popen:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for chunk in iter(lambda: proc.stdout.read(1), ""):
        log_file.write(chunk)
        log_file.flush()
        print(chunk, end="", flush=True)
    proc.wait()
    return proc


def stream_process_pty(cmd: List[str], cwd: Path, env: Dict[str, str], log_file) -> subprocess.Popen:
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    try:
        while True:
            try:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in ready:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    log_file.write(data.decode("utf-8", errors="replace"))
                    log_file.flush()
                    write_stdout_bytes(data)
                elif proc.poll() is not None:
                    break
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
    finally:
        os.close(master_fd)
    proc.wait()
    return proc


def infer_num_tensors(model_path: str, tensors_per_layer: Optional[int]) -> int:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not infer model structure. Pass --num_tensors explicitly."
        ) from exc

    model_type = getattr(cfg, "model_type", "")
    if tensors_per_layer is None:
        if model_type in {"llama", "mistral", "qwen2", "qwen3"}:
            tensors_per_layer = 7
        elif model_type == "opt":
            tensors_per_layer = 6
        elif model_type == "falcon":
            tensors_per_layer = 4
        else:
            raise ValueError(
                f"Unknown model_type={model_type}; pass --tensors_per_layer or --num_tensors."
            )

    num_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
    )
    if num_layers is None:
        raise ValueError("Could not infer number of layers; pass --num_tensors.")
    return int(num_layers) * int(tensors_per_layer)


def infer_example_model_key(model_path: str) -> str:
    text = model_path.lower().replace("_", "-")
    if "llama-2" in text or "llama2" in text:
        return "llama2_7b"
    if "llama-3" in text or "llama3" in text:
        return "llama3_70b" if "70b" in text else "llama3_8b"
    if "mistral" in text:
        return "mistral_7b"
    if "falcon" in text:
        return "falcon_7b"
    if "opt" in text and ("6.7" in text or "6b7" in text or "6-7" in text):
        return "opt6b7"
    raise ValueError(
        "Could not infer example bit-pattern key from --model_path. "
        "Use a model path containing one of: llama2, falcon, llama3, mistral, opt6b7."
    )


def _eval_bit_expr(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_bit_expr(elt) for elt in node.elts]
    if isinstance(node, ast.BinOp):
        left = _eval_bit_expr(node.left)
        right = _eval_bit_expr(node.right)
        if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
            return left + right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, list) and isinstance(right, int):
                return left * right
            if isinstance(left, int) and isinstance(right, list):
                return left * right
    raise ValueError(f"Unsupported bit-pattern expression: {ast.dump(node)}")


def load_example_bit_pattern(repo_root: Path, method: str, model_key: str) -> List[int]:
    config_path = repo_root / "accel_model_configs_example.py"
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    target_name = f"{method}_cfg"
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == target_name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call) or len(node.value.args) < 2:
            continue
        patterns = node.value.args[1]
        if not isinstance(patterns, ast.Dict):
            continue
        for key_node, value_node in zip(patterns.keys, patterns.values):
            if isinstance(key_node, ast.Constant) and key_node.value == model_key:
                return [int(bit) for bit in _eval_bit_expr(value_node)]
    raise ValueError(f"No initial bit pattern found for method={method}, model={model_key}")


def adapt_initial_bits(bits: List[int], num_tensors: int, method: str, model_key: str) -> List[int]:
    if len(bits) == num_tensors:
        return bits
    if len(bits) > num_tensors:
        important(
            "INITIAL CONFIG",
            f"{method}: truncating {model_key} initial pattern from {len(bits)} to {num_tensors} entries",
            "1;34",
        )
        return bits[:num_tensors]
    repeats = (num_tensors + len(bits) - 1) // len(bits)
    important(
        "INITIAL CONFIG",
        f"{method}: repeating {model_key} initial pattern from {len(bits)} to {num_tensors} entries",
        "1;34",
    )
    return (bits * repeats)[:num_tensors]


def initial_bits_for_method(
    args: argparse.Namespace,
    repo_root: Path,
    method: str,
    num_tensors: int,
) -> List[int]:
    if args.initial_layer_bit_config == "uniform":
        return [args.initial_bit] * num_tensors

    model_key = infer_example_model_key(args.model_path)
    bits = load_example_bit_pattern(repo_root, method, model_key)
    bits = adapt_initial_bits(bits, num_tensors, method, model_key)
    important(
        "INITIAL CONFIG",
        (
            f"{method}: high_bits={sum(bit == args.high_bit for bit in bits)} "
            f"low_bits={sum(bit == args.initial_bit for bit in bits)} "
            f"w_bits={bits_display(bits, args.high_bit)} "
            f"source=accel_model_configs_example.py::{model_key}"
        ),
        "1;34",
    )
    return bits


def write_pattern(path: Path, method: str, bits: List[int], metric_value: Optional[float] = None) -> None:
    payload = {
        "method": method,
        "metric": metric_value,
        "w_bits": bits,
    }
    compact = compact_bits_expr(bits)
    if compact is not None:
        payload["compact_w_bits"] = compact
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pattern_key(bits: Iterable[int]) -> str:
    return ",".join(str(x) for x in bits)


def compact_bits_expr(bits: List[int], tensors_per_layer: int = 7) -> Optional[str]:
    if tensors_per_layer <= 0 or len(bits) % tensors_per_layer != 0:
        return None
    template = bits[:tensors_per_layer]
    repeats = len(bits) // tensors_per_layer
    if template * repeats != bits:
        return None
    return f"[{','.join(str(bit) for bit in template)}]*{repeats}"


def llama3_8b_grouped_search(args: argparse.Namespace, num_tensors: int) -> bool:
    if args.tensors_per_layer not in (None, 7):
        return False
    if num_tensors % 7 != 0:
        return False
    try:
        return infer_example_model_key(args.model_path) == "llama3_8b"
    except ValueError:
        return False


def candidate_changes(
    args: argparse.Namespace,
    bits: List[int],
    from_bit: int,
    limit: int,
    grouped_llama3_8b: bool,
) -> List[Tuple[str, List[int]]]:
    if grouped_llama3_8b:
        num_layers = len(bits) // 7
        groups = []
        for local_idx in range(7):
            indices = [layer * 7 + local_idx for layer in range(num_layers)]
            if any(bits[idx] == from_bit for idx in indices):
                groups.append((f"tensor_local={local_idx}", indices))
        return groups[:limit] if limit > 0 else groups

    indices = candidate_indices(bits, from_bit, limit)
    return [(f"tensor_index={idx}", [idx]) for idx in indices]


def remaining_evals_text(args: argparse.Namespace, completed: int) -> str:
    if args.max_evals <= 0:
        return "unlimited"
    return str(max(args.max_evals - completed, 0))


def direction_for_value(value: float, target: float, kind: str, args: argparse.Namespace) -> str:
    if within_tolerance(value, target, args):
        return "stop"
    return "increase precision" if is_worse(value, target, kind, args) else "decrease precision"


def print_eval_config(
    args: argparse.Namespace,
    method: str,
    eval_id: int,
    bits: List[int],
    kind: str,
    target: float,
    pattern_path: Path,
    log_path: Path,
) -> None:
    payload = {
        "method": method,
        "eval": eval_id,
        "metric": kind,
        "target": target,
        "high_bit": args.high_bit,
        "high_bit_count": sum(bit == args.high_bit for bit in bits),
        "low_bit": args.initial_bit,
        "low_bit_count": sum(bit == args.initial_bit for bit in bits),
        "remaining_after_this": remaining_evals_text(args, eval_id + 1),
        "config": str(pattern_path),
        "log": str(log_path),
        "w_bits": compact_bits_expr(bits) or "mixed",
    }
    section("EVAL CONFIG", f"{method} #{eval_id}", "1;35")
    print(json.dumps(payload, indent=2), flush=True)


def within_tolerance(value: float, target: float, args: argparse.Namespace) -> bool:
    tol = max(args.abs_tolerance, args.rel_tolerance * abs(target))
    return abs(value - target) <= tol


def is_worse(value: float, target: float, kind: str, args: argparse.Namespace) -> bool:
    tol = max(args.abs_tolerance, args.rel_tolerance * abs(target))
    return value > target + tol if lower_is_better(kind) else value < target - tol


def candidate_indices(bits: List[int], from_bit: int, limit: int) -> List[int]:
    indices = [i for i, bit in enumerate(bits) if bit == from_bit]
    if limit > 0:
        return indices[:limit]
    return indices


def method_dtype(args: argparse.Namespace, method: str) -> str:
    return getattr(args, f"{method}_dtype")


def method_group_size(args: argparse.Namespace, method: str) -> int:
    return args.mant_group_size if method == "mant" else args.method_group_size


def evaluate_method(
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    method: str,
    bits: List[int],
    kind: str,
    target: float,
    cache: Dict[str, float],
) -> float:
    key = pattern_key(bits)
    if key in cache:
        return cache[key]

    pattern_path = run_dir / f"{method}_{len(cache):04d}.json"
    write_pattern(pattern_path, method, bits)
    log_path = run_dir / f"{method}_{len(cache):04d}.log"
    eval_id = len(cache)
    print_eval_config(args, method, eval_id, bits, kind, target, pattern_path, log_path)
    quant_bit_width = f"w{args.initial_bit}a{args.a_bit}k{args.k_bit}v{args.v_bit}"
    cmd = [
        sys.executable,
        "-m",
        "run_evaluation",
        "--model_path",
        args.model_path,
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
        "--num_fewshot",
        str(args.num_fewshot),
        "--quant_bit_width",
        quant_bit_width,
        "--quant_mode",
        method,
        "--quant_dtype",
        method_dtype(args, method),
        "--q_group_size",
        str(method_group_size(args, method)),
        "--layer_bit_config",
        str(pattern_path),
        "--layer_a_bits",
        args.layer_a_bits,
    ]
    if args.limit_samples is not None:
        cmd.extend(["--limit_samples", str(args.limit_samples)])
    run_command(cmd, repo_root, log_path, args.dry_run)
    value = 0.0 if args.dry_run else parse_metric_from_log(
        log_path.read_text(encoding="utf-8", errors="ignore"), kind, args.tasks
    )
    cache[key] = value
    write_pattern(pattern_path, method, bits, value)
    important(
        "EVAL RESULT",
        (
            f"method={method} eval={eval_id} {kind}={value} target={target} "
            f"direction_if_selected={direction_for_value(value, target, kind, args)} "
            f"high_bits={sum(bit == args.high_bit for bit in bits)} "
            f"low_bits={sum(bit == args.initial_bit for bit in bits)} "
            f"w_bits={bits_display(bits, args.high_bit)} "
            f"completed_evals={len(cache)} remaining_evals={remaining_evals_text(args, len(cache))}"
        ),
    )
    return value


def evaluate_sota(args: argparse.Namespace, repo_root: Path, run_dir: Path, kind: str) -> float:
    if args.target_metric is not None:
        important("SOTA TARGET", f"using provided nvesm2 {kind}={args.target_metric}", "1;33")
        return args.target_metric

    sota_cwd = (repo_root / args.sota_repo).resolve()
    log_path = run_dir / "sota_nvesm2.log"
    cmd = [
        args.sota_python,
        "-m",
        "mxq.entry",
        "--model_path",
        args.model_path,
        "--tasks",
        args.tasks,
        "--batch_size",
        str(args.batch_size),
        "--num_fewshot",
        str(args.num_fewshot),
        "--w_bit",
        str(args.sota_w_bit),
        "--w_mode",
        args.sota_w_mode,
        "--a_bit",
        str(args.sota_a_bit),
        "--a_mode",
        args.sota_a_mode,
        "--group_size",
        str(args.sota_group_size),
    ]
    if args.sota_limit_samples is not None:
        cmd.extend(["--limit_samples", str(args.sota_limit_samples)])
    run_command(cmd, sota_cwd, log_path, args.dry_run)
    return 0.0 if args.dry_run else parse_metric_from_log(
        log_path.read_text(encoding="utf-8", errors="ignore"), kind, args.tasks
    )


def search_method(
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    method: str,
    target: float,
    kind: str,
    num_tensors: int,
) -> Tuple[List[int], float]:
    bits = initial_bits_for_method(args, repo_root, method, num_tensors)
    grouped_llama3_8b = llama3_8b_grouped_search(args, num_tensors)
    if grouped_llama3_8b:
        important(
            "GROUPED SEARCH",
            f"{method}: llama3_8b grouped search enabled; each trial changes one tensor position across "
            f"{num_tensors // 7} layers",
            "1;34",
        )
    cache: Dict[str, float] = {}
    visited_selected: Set[str] = {pattern_key(bits)}
    eval_count = 0

    for step in range(args.max_steps + 1):
        value = evaluate_method(args, repo_root, run_dir, method, bits, kind, target, cache)
        eval_count = len(cache)
        if within_tolerance(value, target, args):
            important(
                "NEXT",
                f"method={method} step={step} action=stop reason=reached_tolerance "
                f"completed_evals={eval_count} remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            return bits, value
        if args.max_evals > 0 and eval_count >= args.max_evals:
            important(
                "NEXT",
                f"method={method} step={step} action=stop reason=max_evals "
                f"completed_evals={eval_count} remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            return bits, value

        promote = is_worse(value, target, kind, args)
        from_bit, to_bit = (args.initial_bit, args.high_bit) if promote else (args.high_bit, args.initial_bit)
        changes = candidate_changes(args, bits, from_bit, args.max_candidates_per_step, grouped_llama3_8b)
        if not changes:
            important(
                "NEXT",
                f"method={method} step={step} action={'increase' if promote else 'decrease'}_precision "
                f"from_bit={from_bit} to_bit={to_bit} candidates=0 reason=no_candidates "
                f"completed_evals={eval_count} remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            return bits, value
        candidate = bits.copy()
        candidate_items = []
        skipped_cycle_labels = []
        for label, change_indices in changes:
            for idx in change_indices:
                candidate[idx] = to_bit
            candidate_key = pattern_key(candidate)
            if candidate_key in visited_selected:
                skipped_cycle_labels.append(label)
                continue
            candidate_items.append((label, change_indices, candidate.copy()))
        if not candidate_items:
            important(
                "NEXT",
                f"method={method} step={step} action=stop reason=no_unvisited_candidates "
                f"from_bit={from_bit} to_bit={to_bit} skipped_cycle_candidates={skipped_cycle_labels} "
                f"visited_selected={len(visited_selected)} completed_evals={eval_count} "
                f"remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            return bits, value
        label, change_indices, candidate_bits = candidate_items[0]
        important(
            "NEXT",
            f"method={method} step={step} action={'increase' if promote else 'decrease'}_precision "
            f"from_bit={from_bit} to_bit={to_bit} "
            f"candidate={label} "
            f"queued_candidates={[item_label for item_label, _, _ in candidate_items]} "
            f"skipped_cycle_candidates={skipped_cycle_labels} "
            f"completed_evals={eval_count} "
            f"remaining_evals={remaining_evals_text(args, eval_count)}",
            "1;33",
        )

        section(
            "CANDIDATE",
            f"method={method} step={step} "
            f"{label} changed_indices={change_indices} change={from_bit}->{to_bit}",
            "1;36",
        )
        candidate_value = evaluate_method(args, repo_root, run_dir, method, candidate_bits, kind, target, cache)
        eval_count = len(cache)
        if within_tolerance(candidate_value, target, args):
            important(
                "NEXT",
                f"method={method} step={step} action=stop reason=reached_tolerance "
                f"candidate={label} completed_evals={eval_count} "
                f"remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            return candidate_bits, candidate_value

        bits = candidate_bits
        visited_selected.add(pattern_key(bits))
        important(
            "SELECT",
            f"method={method} step={step} selected_metric={candidate_value} "
            f"high_bits={sum(bit == args.high_bit for bit in bits)} "
            f"low_bits={sum(bit == args.initial_bit for bit in bits)} "
            f"w_bits={bits_display(bits, args.high_bit)} "
            f"completed_evals={eval_count} remaining_evals={remaining_evals_text(args, eval_count)}",
            "1;32",
        )
        if args.max_evals > 0 and eval_count >= args.max_evals:
            important(
                "NEXT",
                f"method={method} step={step} action=stop reason=max_evals "
                f"completed_evals={eval_count} remaining_evals={remaining_evals_text(args, eval_count)}",
                "1;33",
            )
            break

    final_value = evaluate_method(args, repo_root, run_dir, method, bits, kind, target, cache)
    return bits, final_value


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    kind = metric_kind(args)
    num_tensors = args.num_tensors or infer_num_tensors(args.model_path, args.tensors_per_layer)
    important(
        "SEARCH START",
        (
            f"metric={kind} lower_is_better={lower_is_better(kind)} num_tensors={num_tensors} "
            f"sample={args.limit_samples if args.limit_samples is not None else 'full'} "
            f"sota_sample={args.sota_limit_samples if args.sota_limit_samples is not None else 'full'}"
        ),
        "1;36",
    )

    target = evaluate_sota(args, repo_root, run_dir, kind)
    important("SOTA TARGET", f"nvesm2 {kind}={target}", "1;33")

    summary = {"target": target, "metric": kind, "methods": {}}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        bits, value = search_method(args, repo_root, method_dir, method, target, kind, num_tensors)
        out_path = run_dir / f"final_{method}.json"
        write_pattern(out_path, method, bits, value)
        summary["methods"][method] = {
            "metric": value,
            "high_bit_count": sum(bit == args.high_bit for bit in bits),
            "w_bits": bits_display(bits, args.high_bit),
            "config": str(out_path),
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    section("FINAL SUMMARY", str(run_dir), "1;32")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
