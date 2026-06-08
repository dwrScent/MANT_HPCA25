=== EVAL RESULT === method=ant eval=5 ppl=6.547922611236572 target=6.555898666381836 direction_if_selected=stop high_bits=128 low_bits=96 w_bits=[4,4,8,8,8,4,8]*32 completed_evals=6 remaining_evals=unlimited


=== NEXT === method=ant step=4 action=stop reason=reached_tolerance candidate=tensor_local=1 completed_evals=6 remaining_evals=unlimited


=== FINAL SUMMARY === /root/llm-quan/MANT_HPCA25/output/precision_search/20260512_165454
{
  "target": 6.555898666381836,
  "metric": "ppl",
  "methods": {
    "ant": {
      "metric": 6.547922611236572,
      "high_bit_count": 128,
      "w_bits": "[4,4,8,8,8,4,8]*32",
      "config": "/root/llm-quan/MANT_HPCA25/output/precision_search/20260512_165454/final_ant.json"
    }
  }
}
