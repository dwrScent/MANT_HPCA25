=== EVAL RESULT === method=mant eval=0 ppl=6.570427417755127 target=6.555898666381836 direction_if_selected=stop high_bits=96 low_bits=128 w_bits=[8,4,8,4,4,4,8]*32 completed_evals=1 remaining_evals=unlimited


=== NEXT === method=mant step=0 action=stop reason=reached_tolerance completed_evals=1 remaining_evals=unlimited


=== FINAL SUMMARY === /root/llm-quan/MANT_HPCA25/output/precision_search/20260512_175445
{
  "target": 6.555898666381836,
  "metric": "ppl",
  "methods": {
    "mant": {
      "metric": 6.570427417755127,
      "high_bit_count": 96,
      "w_bits": "[8,4,8,4,4,4,8]*32",
      "config": "/root/llm-quan/MANT_HPCA25/output/precision_search/20260512_175445/final_mant.json"
    }
  }
}


