# MANT Downstream Evaluation Result

- Run id: 20260602_111246
- Updated at: 2026-06-02 18:09:14
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
| meta-flint | 72.23 | 77.37 | 67.32 | 78.66 |  |  | 73.90 |
| fp16 |  |  |  |  |  |  |  |

#### Accuracy Norm

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 69.14 | 74.27 |  | 72.81 | 47.95 |  | 66.04 |
| ant | 70.84 | 74.54 |  | 75.97 | 51.96 |  | 68.33 |
| mant | 70.38 | 75.30 |  | 74.41 | 51.62 |  | 67.93 |
| meta-flint | 72.23 | 77.37 |  | 78.66 |  |  | 76.09 |
| fp16 |  |  |  |  |  |  |  |

### llama3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive |  |  |  |  |  |  |  |
| ant |  |  |  |  |  |  |  |
| mant |  |  |  |  |  |  |  |
| meta-flint |  |  |  |  |  |  |  |
| fp16 |  |  |  |  |  |  |  |

## Group Size 32

### qwen3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive |  |  |  |  |  |  |  |
| ant |  |  |  |  |  |  |  |
| mant |  |  |  |  |  |  |  |
| meta-flint |  |  |  |  |  |  |  |
| fp16 |  |  |  |  |  |  |  |

### llama3-8b

#### Primary Metric

| Method | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olive |  |  |  |  |  |  |  |
| ant |  |  |  |  |  |  |  |
| mant |  |  |  |  |  |  |  |
| meta-flint |  |  |  |  |  |  |  |
| fp16 |  |  |  |  |  |  |  |
