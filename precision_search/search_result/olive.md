# Closest SOTA Results

Target: nvesm2 W4A4, `wikitext`, `limit_samples=64`, `ppl=6.555898666381836`.
Lower PPL is better.

| method | selected result | ppl | delta vs target | abs delta | high bits | low bits | compact w_bits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| olive | `output/precision_search/20260512_114723/olive/olive_0010.json` | 6.547967910766602 | -0.007930755615234375 | 0.007930755615234375 | 160 | 64 | `[8,8,4,4,8,8,8]*32` |

