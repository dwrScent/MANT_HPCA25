# MANT Downstream Evaluation Result

- Run id: 20260602_111246
- Updated at: 2026-06-03 17:13:35
- Methods: olive, ant, mant, meta-flint, fp16
- Default quant bit width: w4a4k16v16
- Group sizes: 64, 32
- Few-shot: 0
- Default batch size: 32
- BoolQ batch size: 8
- ReCoRD batch size: 1
- Models: qwen3-8b, llama3-8b
- Tasks: hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq
- Log dir: /root/llm-quan/MANT_HPCA25/output/auto_logs/20260602_111246

## Group Size 64

### qwen3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 69.14 | 74.27 | 62.83 | 72.81 | 47.95 | 82.39 | 68.23 |
| ant | 70.84 | 74.54 | 64.96 | 75.97 | 51.96 | 85.38 | 70.61 |
| mant | 70.38 | 75.30 | 61.88 | 74.41 | 51.62 | 84.04 | 69.61 |
| meta-flint | 72.23 | 77.37 | 67.32 | 78.66 | 54.44 | 85.81 | 72.64 |
| fp16 | 74.97 | 77.80 | 67.72 | 80.85 | 56.48 | 86.57 | 74.06 |

#### Accuracy Norm

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 69.14 | 74.27 |  | 72.81 | 47.95 |  | 66.04 |
| ant | 70.84 | 74.54 |  | 75.97 | 51.96 |  | 68.33 |
| mant | 70.38 | 75.30 |  | 74.41 | 51.62 |  | 67.93 |
| meta-flint | 72.23 | 77.37 |  | 78.66 | 54.44 |  | 70.68 |
| fp16 | 74.97 | 77.80 |  | 80.85 | 56.48 |  | 72.52 |

### llama3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 73.53 | 75.63 | 64.40 | 70.41 | 43.86 | 74.10 | 66.99 |
| ant | 74.51 | 76.39 | 67.25 | 70.79 | 45.65 | 74.80 | 68.23 |
| mant | 75.90 | 77.97 | 68.51 | 73.82 | 49.40 | 76.88 | 70.41 |
| meta-flint | 76.36 | 79.65 | 70.80 | 75.51 | 48.72 | 79.48 | 71.75 |
| fp16 | 79.15 | 80.79 | 72.77 | 77.65 | 53.24 | 81.38 | 74.16 |

#### Accuracy Norm

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 73.53 | 75.63 |  | 70.41 | 43.86 |  | 65.86 |
| ant | 74.51 | 76.39 |  | 70.79 | 45.65 |  | 66.83 |
| mant | 75.90 | 77.97 |  | 73.82 | 49.40 |  | 69.27 |
| meta-flint | 76.36 | 79.65 |  | 75.51 | 48.72 |  | 70.06 |
| fp16 | 79.15 | 80.79 |  | 77.65 | 53.24 |  | 72.71 |

## Group Size 32

### qwen3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 65.77 | 73.34 | 63.93 | 75.97 | 47.53 | 83.73 | 68.38 |
| ant | 70.87 | 75.24 | 65.90 | 75.97 | 51.02 | 83.94 | 70.49 |
| mant | 71.94 | 76.01 | 66.30 | 76.89 | 52.65 | 85.23 | 71.50 |
| meta-flint | 72.81 | 76.61 | 66.77 | 77.36 | 53.84 | 85.26 | 72.11 |
| fp16 | 74.97 | 77.80 | 67.72 | 80.85 | 56.48 | FAIL | 71.56 |

#### Accuracy Norm

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 65.77 | 73.34 |  | 75.97 | 47.53 |  | 65.65 |
| ant | 70.87 | 75.24 |  | 75.97 | 51.02 |  | 68.28 |
| mant | 71.94 | 76.01 |  | 76.89 | 52.65 |  | 69.37 |
| meta-flint | 72.81 | 76.61 |  | 77.36 | 53.84 |  | 70.16 |
| fp16 | 74.97 | 77.80 |  | 80.85 | 56.48 | FAIL | 72.52 |

### llama3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 67.52 | 72.69 | 65.11 | 63.43 | 37.29 | 67.58 | 62.27 |
| ant | 74.85 | 77.09 | 68.75 | 72.47 | 46.50 | 75.63 | 69.21 |
| mant | FAIL | FAIL | FAIL | 75.17 | 51.02 | 78.56 | 68.25 |
| meta-flint | 77.02 | 78.73 | 69.85 | 75.42 | 50.43 | 77.92 | 71.56 |
| fp16 | 79.15 | 80.79 | 72.77 | 77.65 | 53.24 | 81.38 | 74.16 |

#### Accuracy Norm

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 67.52 | 72.69 |  | 63.43 | 37.29 |  | 60.23 |
| ant | 74.85 | 77.09 |  | 72.47 | 46.50 |  | 67.73 |
| mant | FAIL | FAIL | FAIL | 75.17 | 51.02 |  | 63.09 |
| meta-flint | 77.02 | 78.73 |  | 75.42 | 50.43 |  | 70.40 |
| fp16 | 79.15 | 80.79 |  | 77.65 | 53.24 |  | 72.71 |

## Failed or Skipped Tasks

| Group | Model | Method | Task | Status | Log |
| --- | --- | --- | --- | --- | --- |
| 32 | qwen3-8b | fp16 | boolq | FAIL | 20260602_111246_qwen3-8b_g32_fp16_boolq.log |
| 32 | llama3-8b | mant | hellaswag | FAIL | 20260602_111246_llama3-8b_g32_mant_hellaswag.log |
| 32 | llama3-8b | mant | piqa | FAIL | 20260602_111246_llama3-8b_g32_mant_piqa.log |
| 32 | llama3-8b | mant | winogrande | FAIL | 20260602_111246_llama3-8b_g32_mant_winogrande.log |
