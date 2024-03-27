# Datasets

### Directory Structure

```
├── llama           # Tokenized version of datasets for Llama 13B models
├── llama-7         # Tokenized version of datasets for Llama 7B models
├── opt             # Tokenized version of datasets for OPT 13B models
├── opt-6.7         # Tokenized version of datasets for OPT 6.7B models
├── raw             # Raw datasets in JSON (Lines) format
└── scripts         # Scripts to download and prepare datasets
```

### Dataset Sources

- [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [ShareGPT](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)
- [CNN DailyMail](https://huggingface.co/datasets/ccdv/cnn_dailymail)
- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

### Dataset Overview

#### Alpaca
Alpaca is a dataset of 52k instructions and demonstrations. It contains a variety
of different tasks demonstrated [here](https://github.com/tatsu-lab/stanford_alpaca/blob/main/assets/parse_analysis.png).
The majority of the tasks generate text of different sizes as output. Thus, we treat
Alpaca dataset as a **text generation** one.

#### ShareGPT
ShareGPT is a dataset collected from [ShareGPT](https://sharegpt.com) in March 2023.
It contains multi-round conversations between humans and ChatGPT. We consider this
as our **conversational** dataset, but we also tranform it into a text completion dataset
by extracting the first round of each conversation (the user's first input and ChatGPT's
first response).

#### CNN DailyMail
CNN DailyMail is a dataset that contains 300k article and their highlights. We treat
this dataset as a **text summarization** dataset. The merged version of the highlights
of an article is essentially its summary.

#### Dolly
Dolly is a dataset of 15k instruction-following records classified in behavioral
categories outlined in the [InstructGPT paper](https://arxiv.org/abs/2203.02155),
including brainstorming, classification, closed QA, generation, information extraction,
open QA, and summarization. We only use the following 3 categories (~60% of the 
dataset): closedQA, openQA and generalQA, thus making this our **question-answering**
dataset.

### Download and Preparation Instructions

```
pip install datasets
```
Before running the following scripts, change `HF_TOKEN` parameter in `scripts/utils.py`.

```
cd datasets
python scripts/download.py
python scripts/prepare.py    
python scripts/prepare_chatbot.py
```

### Prepared Dataset Structure

Every folder that contains the tokenized version of the datasets has the following
structure:

```
.
├── alpaca_data.pkl             # Tokenized version of Alpaca dataset
├── cnn_dailymail_data.pkl      # Tokenized version of CNN DailyMail dataset
├── dolly_data.pkl              # Tokenized version of Dolly dataset
├── sharegpt_chat_data.pkl      # Tokenized version of ShareGPT dataset (chat version)
└── sharegpt_data.pkl           # Tokenized version of ShareGPT dataset
```

Each of the files, except `sharegpt_chat_data.pkl`, has the following format

```py
[(input_ids, output_ids), ...]: List[Tuple[List[int], List[int]]]
```

`input_ids`: List of the token ids representing the input sequence

`output_ids`: List of the token ids representing the output sequence

---
`sharegpt_chat_data.pkl`: Since the original ShareGPT dataset contains multi-round
conversations, we provide the following format for the chatbot benchmark scenario:

```py
[[("human", input_ids), ("gpt", output_ids), ...], ...]: List[List[Tuple[["human", "gpt"], List[int]]]]
```
