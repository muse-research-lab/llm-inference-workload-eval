# LLM Inference Workload Evaluation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains all the code used for the experimental analysis of our paper:

[The Importance of Workload Choice in Evaluating LLM Inference Systems](https://doi.org/10.1145/3642970.3655823).

This paper presents an extensive experimental evaluation that aims to capture the impact of the workload used for evaluation and quantify the benefit derived from higher memory availability.
Our analysis shows that LLMs can achieve 3 $\times$ higher throughput for text generation and question-answering use cases compared to text summarization and conversational ones.
The latter ones seem to exhibit low levels of performance due to their demanding input sizes.
In addition, non-latency-critical inference services achieve 2.3 $\times$ higher throughput when 4 $\times$ more memory is available.
In conclusion, this paper aims to highlight the importance and impact of the chosen workloads in the evaluation of systems for LLM inference.

## Description

```
├── datasets        # Datasets used for the evaluation
├── experiments     # Scripts for the experiments
├── figures         # Figures of the paper
├── proc-outputs    # Processed outputs used for the analysis
├── results         # Raw experimental results
├── vllm            # Extended version of vLLM
├── analysis.ipynb  # Experimental analysis
├── metadata.py     # Metadata for the experimental analysis
└── utils.py        # Utils for the experimental analysis
```

## Hardware Requirements
- NVIDIA GPU A100 40GB

## Software Requirements
- Python >= 3.11
- CUDA 11.7

You also need to have a Hugging Face token with access to LLama-2 models!

## Installation

Clone the repository:

```bash
git clone --recursive https://github.com/muse-research-lab/llm-inference-workload-eval.git
```

Set up environment:

```bash
conda create -n vllm python=3.11.2
```

```bash
conda activate vllm
```

Install vLLM package and its dependencies:

```bash
cd vllm && pip install -r requirements.txt
```

```bash
pip install -e .
```

### CUDA Configuration

In case you face a CUDA version mismatch error while installing the package:

1. Download the required CUDA version [here](https://developer.nvidia.com/cuda-downloads).
2. Activate the appropraite CUDA Compiler Driver NVCC

```bash
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
```

## Dataset Preparation

To prepare the datasets, follow the instructions of the [README](datasets/README.md) file under the `datasets` directory.

## Experiments

Figure 2:
```
bash experiments/online-643.sh
bash experiments/offline-643.sh
```

## Analysis

Go through the `analysis.ipynb` to produce the figures of the paper.
In case you do not want to run the experiments of the previous section, you can skip the "Preprocess" section of "Figure 2" and directly load the results from the "Load" section.
The produced figures will be stored under the `figures` directory.

## Citation

```bibtex
@inproceedings{,
    author = {Papaioannou, Konstantinos and Doudali, Thaleia Dimitra},
    title = {The Importance of Workload Choice in Evaluating LLM Inference Systems},
    year = {2024},
    isbn = {},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {},
    doi = {},
    booktitle = {Proceedings of the 4th Workshop on Machine Learning and Systems},
    pages = {},
    numpages = {},
    location = {Athens, Greece},
    series = {EuroMLSys '24}
}
```

## Acknowledgement

This work is part of the grants FJC2021-047102-I, TED2021-132464B-I00, PID2022-142290OB-I00, funded by the European Union «NextGenerationEU»/PRTR, the ESF+ and MCIN/AEI/10.13039/501100011033.

![Acknowledgement](https://raw.githubusercontent.com/muse-research-lab/cloud-traces-comparison/main/docs/images/acknowledgement.png)
