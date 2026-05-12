=== TRIAL === method=mant step=0 trial=1/5 tensor_local=0 changed_indices=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217] change=4->8

=== EVAL CONFIG === mant #1
{
  "method": "mant",
  "eval": 1,
  "metric": "ppl",
  "target": 6.555898666381836,
  "high_bit": 8,
  "high_bit_count": 96,
  "low_bit": 4,
  "low_bit_count": 128,
  "remaining_after_this": "unlimited",
  "config": "/root/llm-quan/MANT_HPCA25/output/precision_search/20260512_103049/mant/mant_0001.json",
  "log": "/root/llm-quan/MANT_HPCA25/output/precision_search/20260512_103049/mant/mant_0001.log",
  "w_bits": "[8,4,8,4,4,4,8]*32"
}

=== EVAL RESULT === method=mant eval=1 ppl=6.570427417755127 target=6.555898666381836 direction_if_selected=stop high_bits=96 low_bits=128 w_bits=[8,4,8,4,4,4,8]*32 completed_evals=2 remaining_evals=unlimited
