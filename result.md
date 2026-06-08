# MANT Downstream Evaluation Result

- Run id: 20260608_205841
- Updated at: 2026-06-08 22:35:43
- Methods: olive, ant, mant, meta-flint, fp16
- Default quant bit width: w4a4k16v16
- Group sizes: 64, 32
- Few-shot: 0
- Default batch size: 32
- BoolQ batch size: 8
- ReCoRD batch size: 1
- Models: qwen3-8b, llama3-8b
- Tasks: mmlu, gsm8k, hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq
- GSM8K accuracy uses exact_match.
- Log dir: /root/llm-quan/MANT_HPCA25/output/auto_logs/20260608_205841

## Group Size 64

### qwen3-8b

#### Accuracy

| Method | mmlu | gsm8k | hellaswag | piqa | winogrande | arc_easy | arc_challenge | boolq | avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olive | 64.74 |  |  |  |  |  |  |  | 64.74 |
| ant |  |  |  |  |  |  |  |  |  |
| mant |  |  |  |  |  |  |  |  |  |
| meta-flint |  |  |  |  |  |  |  |  |  |
| fp16 |  |  |  |  |  |  |  |  |  |

### llama3-8b

## Group Size 32

### qwen3-8b

### llama3-8b
