### Mixed-precision search against nvesm2

All precision-search defaults are set inside `scripts/run_precision_search.sh`.
The command line only needs the nvesm2 target metric:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_precision_search.sh TARGET_METRIC
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_precision_search.sh 5.87
```

The script searches ANT, OliVe, and M-ANT per-linear weight bit-widths against the
given nvesm2 target. By default it uses:

- `MODEL_PATH=/cephfs/shared/model/llama-3-8b-hf`
- `TASKS=wikitext`
- `SHOTS=0`
- `METHODS=olive,ant,mant`
- `SAMPLE=200`
- `INITIAL_MODEL_LAYER_CONFIG=example`
- `BATCH_SIZE=32`
- `LAYER_A_BITS=follow`
- `MAX_CANDIDATES_PER_STEP=8`

Edit these defaults in `scripts/run_precision_search.sh`, or override them with
environment variables:

```bash
MODEL_PATH=/path/to/model TASKS=c4 METHODS=ant,mant \
CUDA_VISIBLE_DEVICES=0 ./scripts/run_precision_search.sh 6.12
```

Outputs are written under `output/precision_search/<run_id>/`. Use the generated
`final_<method>.json` as `--layer_bit_config` for final evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m run_evaluation \
  --model_path /path/to/model \
  --tasks wikitext \
  --quant_mode ant \
  --quant_dtype int-flint-pot-float \
  --q_group_size 64 \
  --quant_bit_width w4a4k16v16 \
  --layer_bit_config output/precision_search/<run_id>/final_ant.json \
  --layer_a_bits follow
```

