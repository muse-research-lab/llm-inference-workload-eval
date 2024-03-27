import os

from transformers import AutoTokenizer, LlamaTokenizer

RAW_DATASETS_PATH = os.path.abspath(os.path.join(os.getcwd(), "raw"))

DATASETS_META = {
    "tatsu-lab/alpaca": {
        "version": None,
        "raw_file_name": "alpaca_data.json",
        "dataset_file_name": "alpaca_data.pkl",
    },
    "Aeala/ShareGPT_Vicuna_unfiltered": {
        "version": None,
        "raw_file_name": "sharegpt_data.json",
        "dataset_file_name": "sharegpt_data.pkl",
        "chat_dataset_file_name": "sharegpt_chat_data.pkl",
    },
    "ccdv/cnn_dailymail": {
        "version": "3.0.0",
        "raw_file_name": "cnn_dailymail_data.json",
        "dataset_file_name": "cnn_dailymail_data.pkl",
    },
    "databricks/databricks-dolly-15k": {
        "version": None,
        "raw_file_name": "dolly_data.json",
        "dataset_file_name": "dolly_data.pkl",
    },
}

HF_TOKEN = "hf_"

TOKENIZERS = {
    "llama": LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", token=HF_TOKEN),
    "llama-7": AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=HF_TOKEN),
    "opt": AutoTokenizer.from_pretrained("facebook/opt-13b"),
    "opt-6.7": AutoTokenizer.from_pretrained("facebook/opt-6.7b"),
}