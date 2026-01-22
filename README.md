# M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type

A lightweight pseudo-quantization framework for transformer models, implementing M-ANT, ANT, and OliVe quantization with configurable bit-widths and grouping strategies.

## Setup

1.	Create and activate a conda environment:

```shell
git clone https://github.com/SJTU-ReArch-Group/MANT_HPCA25
cd MANT_HPCA25

conda create -n mant python=3.10 -y
conda activate mant
```

2.	Install the package in development mode:
```shell
pip install --upgrade pip 
pip install -e .
```

## Usage

Run evaluation with pseudo-quantization (simulated quantization + dequantization) for LLaMA/OPT on multiple tasks.

```bash
# Change the model_path inside scripts/llama2_run.sh based on your local path.

# ANT, OliVe (channel-/tensor-wise)
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 ant int-flint-pot-float -1 w4a4k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 olive int-flint -1 w4a4k16v16

# M-ANT
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mant int 64 w4a8k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 mant int 64 w4a8k4v4
# The search over `a` takes about 20 minutes. We recommend using `dump`
# to save the quantized weights on the first run and `load` to reuse them.

# ANT, OliVe (group-wise)
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 ant int-flint-pot-float 64 w4a4k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 olive int-flint 64 w4a4k16v16

# Evaluation on C4
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 c4 0 ant int-flint-pot-float -1 w4a4k16v16

# 8-bit baselines
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 ant int -1 w8a8k16v16
CUDA_VISIBLE_DEVICES=0 ./scripts/llama2_run.sh 7 wikitext 0 olive int -1 w8a8k16v16

```

## Citation

If you find this repository useful in your research or project, please kindly cite:

```text
@inproceedings{hu2025mant,
author={Hu, Weiming and Zhang, Haoyan and Guo, Cong and Feng, Yu and Guan, Renyang and Hua, Zhendong and Liu, Zihan and Guan, Yue and Guo, Minyi and Leng, Jingwen},
booktitle={2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)}, 
title={M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type}, 
year={2025},
pages={1112-1126},
doi={10.1109/HPCA61900.2025.00086}
}

@inproceedings{guo2023olive,
author = {Guo, Cong and Tang, Jiaming and Hu, Weiming and Leng, Jingwen and Zhang, Chen and Yang, Fan and Liu, Yunxin and Guo, Minyi and Zhu, Yuhao},
title = {OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization},
year = {2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3579371.3589038},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
numpages = {15},
location = {Orlando, FL, USA},
series = {ISCA '23}
}

@inproceedings{guo2022ant,
  title={ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization},
  author={Guo, Cong and Zhang, Chen and Leng, Jingwen and Liu, Zihan and Yang, Fan and Liu, Yunxin and Guo, Minyi and Zhu, Yuhao},
  booktitle={2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  pages={1414--1433},
  year={2022},
  organization={IEEE},
  doi={10.1109/MICRO56248.2022.00095}
}
```

## Acknowledgements

+ ANT / OliVe reference implementation: https://github.com/clevercool/ANT-Quantization
+ Code structure inspired by: https://github.com/mit-han-lab/llm-awq
+ Evaluation references: 
  + GPTQ: https://github.com/IST-DASLab/gptq
  + KIVI: https://github.com/jy-yuan/KIVI
