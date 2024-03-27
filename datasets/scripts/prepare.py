import json
import os
import pickle

from datasets import Dataset, load_dataset

from utils import DATASETS_META, RAW_DATASETS_PATH, TOKENIZERS

# Alpaca Dataset Preparation
ALPACA_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH, DATASETS_META["tatsu-lab/alpaca"]["raw_file_name"]
)
def preprocess_alpaca(row):
    row["input"] = row["instruction"] + row["input"]
    return row

print(f"Preprocessing Alpaca dataset...")
alpaca_data = load_dataset("json", data_files=ALPACA_RAW_PATH, split="train"). \
    map(preprocess_alpaca). \
    remove_columns(["instruction", "text"])
print("Finished.")

# ShareGPT Dataset Preparation
SHAREGPT_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH, DATASETS_META["Aeala/ShareGPT_Vicuna_unfiltered"]["raw_file_name"]
)

print(f"Preprocessing ShareGPT text completion dataset...")
sharegpt_tc_data = []
with open(SHAREGPT_RAW_PATH) as f:
    for line in f:
        sharegpt_tc_data.append(json.loads(line))

for idx, data in enumerate(sharegpt_tc_data):
    # TODO: Make sure index 0 is human and 1 gpt
    sharegpt_tc_data[idx]["input"] = data["conversations"][0]["value"]
    sharegpt_tc_data[idx]["output"] = data["conversations"][1]["value"]

    del sharegpt_tc_data[idx]["id"]
    del sharegpt_tc_data[idx]["conversations"]

sharegpt_tc_data = Dataset.from_list(sharegpt_tc_data)
print("Finished.")

# CNN DailyMail Dataset Preparation
CNN_DAILYMAIL_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH, DATASETS_META["ccdv/cnn_dailymail"]["raw_file_name"]
)

print(f"Preprocessing CNN DailyMaily dataset...")
cnn_dailymail_data = load_dataset("json", data_files=CNN_DAILYMAIL_RAW_PATH, split="train"). \
    rename_column("article", "input"). \
    rename_column("highlights", "output"). \
    remove_columns(["id"])
print("Finished.")

# Dolly Dataset Preparation
DOLLY_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH, DATASETS_META["databricks/databricks-dolly-15k"]["raw_file_name"]
)

def preprocess_dolly(row):
    row["input"] = row["instruction"] + row["context"]
    return row

print(f"Preprocessing Dolly dataset...")
dolly_data = load_dataset("json", data_files=DOLLY_RAW_PATH, split="train"). \
    filter(lambda x: "qa" in x["category"]). \
    map(preprocess_dolly). \
    rename_column("response", "output"). \
    remove_columns(["category", "context", "instruction"])
print("Finished.")

# Tokenize Datasets
def encode(row):
    row["input_ids"] = tokenizer.encode(row["input"])
    row["output_ids"] = tokenizer.encode(row["output"])
    return row

def max_len_seqs(row):
    max_len = tokenizer.model_max_length
    return len(row["input"]) < max_len  and len(row["output"]) < max_len

DATASETS = [alpaca_data, sharegpt_tc_data, cnn_dailymail_data, dolly_data]

for name, tokenizer in TOKENIZERS.items():
    print(f"Using {name} tokenizer.")
    path = os.path.abspath(os.path.join(os.getcwd(), name))

    if not os.path.exists(path):
        os.makedirs(path)

    for dataset, meta in zip(DATASETS, DATASETS_META.values()):
        final_path = os.path.join(path, meta["dataset_file_name"])

        print(f"Preparing {meta['dataset_file_name']}...")
        dataset = dataset.filter(max_len_seqs).\
            map(encode). \
            remove_columns(["input", "output"])
        
        saved_dataset = []
        for data in dataset:
            saved_dataset.append((data["input_ids"], data["output_ids"]))
        
        print(f"Saving {meta['dataset_file_name']}...")
        with open(final_path, 'wb') as f:
            pickle.dump(saved_dataset, f)
        print("Finished.")