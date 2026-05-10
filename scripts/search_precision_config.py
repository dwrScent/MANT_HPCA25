#!/usr/bin/env python3
"""Search mixed-precision bit patterns against an nvesm2 target metric."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PPL_TASKS = {"wikitext", "c4", "ptb"}
DEFAULT_METHOD_DTYPES = {
    "ant": "int-flint-pot-float",
    "olive": "int-flint",
    "mant": "int",
}


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
    parser.add_argument("--num_tensors", type=int, default=None)
    parser.add_argument("--tensors_per_layer", type=int, default=None)
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
    parser.add_argument("--sota_w_bit", type=int, default=4)
    parser.add_argument("--sota_a_bit", type=int, default=4)
    parser.add_argument("--sota_w_mode", default="nvesm2")
    parser.add_argument("--sota_a_mode", default="nvesm2")
    parser.add_argument("--sota_group_size", type=int, default=16)
    parser.add_argument("--output_dir", default="output/precision_search")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


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
    print(f"RUN cwd={cwd}: {printable}")
    if dry_run:
        log_file.write_text(printable + "\n", encoding="utf-8")
        return 0.0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:])
        raise RuntimeError(f"Command failed with exit code {proc.returncode}. Log: {log_file}\n{tail}")
    return proc.returncode


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


def write_pattern(path: Path, method: str, bits: List[int], metric_value: Optional[float] = None) -> None:
    payload = {
        "method": method,
        "metric": metric_value,
        "w_bits": bits,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pattern_key(bits: Iterable[int]) -> str:
    return ",".join(str(x) for x in bits)


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
    cache: Dict[str, float],
) -> float:
    key = pattern_key(bits)
    if key in cache:
        return cache[key]

    pattern_path = run_dir / f"{method}_{len(cache):04d}.json"
    write_pattern(pattern_path, method, bits)
    log_path = run_dir / f"{method}_{len(cache):04d}.log"
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
    run_command(cmd, repo_root, log_path, args.dry_run)
    value = 0.0 if args.dry_run else parse_metric_from_log(
        log_path.read_text(encoding="utf-8", errors="ignore"), kind, args.tasks
    )
    cache[key] = value
    write_pattern(pattern_path, method, bits, value)
    print(f"{method}: metric={value}  high_bits={sum(bit == args.high_bit for bit in bits)}")
    return value


def evaluate_sota(args: argparse.Namespace, repo_root: Path, run_dir: Path, kind: str) -> float:
    if args.target_metric is not None:
        print(f"Using provided nvesm2 target metric: {args.target_metric}")
        return args.target_metric

    sota_cwd = (repo_root / args.sota_repo).resolve()
    log_path = run_dir / "sota_nvesm2.log"
    cmd = [
        sys.executable,
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
    run_command(cmd, sota_cwd, log_path, args.dry_run)
    return 0.0 if args.dry_run else parse_metric_from_log(
        log_path.read_text(encoding="utf-8", errors="ignore"), kind, args.tasks
    )


def choose_candidate(
    current_value: float,
    target: float,
    kind: str,
    candidates: List[Tuple[List[int], float]],
) -> Tuple[List[int], float]:
    if lower_is_better(kind):
        if current_value > target:
            return min(candidates, key=lambda item: (abs(item[1] - target), item[1]))
        return min(candidates, key=lambda item: (abs(item[1] - target), -item[1]))
    if current_value < target:
        return min(candidates, key=lambda item: (abs(item[1] - target), -item[1]))
    return min(candidates, key=lambda item: (abs(item[1] - target), item[1]))


def search_method(
    args: argparse.Namespace,
    repo_root: Path,
    run_dir: Path,
    method: str,
    target: float,
    kind: str,
    num_tensors: int,
) -> Tuple[List[int], float]:
    bits = [args.initial_bit] * num_tensors
    cache: Dict[str, float] = {}
    eval_count = 0

    for step in range(args.max_steps + 1):
        value = evaluate_method(args, repo_root, run_dir, method, bits, kind, cache)
        eval_count = len(cache)
        if within_tolerance(value, target, args):
            print(f"{method}: reached tolerance at step {step}")
            return bits, value

        promote = is_worse(value, target, kind, args)
        from_bit, to_bit = (args.initial_bit, args.high_bit) if promote else (args.high_bit, args.initial_bit)
        indices = candidate_indices(bits, from_bit, args.max_candidates_per_step)
        if not indices:
            print(f"{method}: no more {'promotion' if promote else 'demotion'} candidates")
            return bits, value

        candidates = []
        for idx in indices:
            trial = bits.copy()
            trial[idx] = to_bit
            trial_value = evaluate_method(args, repo_root, run_dir, method, trial, kind, cache)
            candidates.append((trial, trial_value))
            eval_count = len(cache)
            if args.max_evals > 0 and eval_count >= args.max_evals:
                break

        new_bits, new_value = choose_candidate(value, target, kind, candidates)
        if pattern_key(new_bits) == pattern_key(bits):
            return bits, value
        bits = new_bits
        print(f"{method}: step={step} selected metric={new_value}")
        if args.max_evals > 0 and eval_count >= args.max_evals:
            break

    final_value = evaluate_method(args, repo_root, run_dir, method, bits, kind, cache)
    return bits, final_value


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    kind = metric_kind(args)
    num_tensors = args.num_tensors or infer_num_tensors(args.model_path, args.tensors_per_layer)
    print(f"metric={kind}, lower_is_better={lower_is_better(kind)}, num_tensors={num_tensors}")

    target = evaluate_sota(args, repo_root, run_dir, kind)
    print(f"nvesm2 target {kind}: {target}")

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
            "config": str(out_path),
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
