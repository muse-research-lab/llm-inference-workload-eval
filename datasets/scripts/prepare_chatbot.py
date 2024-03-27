import json
import os
import pickle

from datasets import Dataset, load_dataset
from tqdm import tqdm

from utils import DATASETS_META, RAW_DATASETS_PATH, TOKENIZERS

# ShareGPT Dataset Preparation
SHAREGPT_RAW_PATH = os.path.join(
    RAW_DATASETS_PATH, DATASETS_META["Aeala/ShareGPT_Vicuna_unfiltered"]["raw_file_name"]
)

print(f"Preprocessing ShareGPT chat dataset...")
sharegpt_chat_data = []
with open(SHAREGPT_RAW_PATH) as f:
    for line in f:
        sharegpt_chat_data.append(json.loads(line))

for idx, data in enumerate(sharegpt_chat_data):
    sharegpt_chat_data[idx]["input"] = []
    for line in data["conversations"]:
        sharegpt_chat_data[idx]["input"].append((line["from"], line["value"]))

    del sharegpt_chat_data[idx]["id"]
    del sharegpt_chat_data[idx]["conversations"]

sharegpt_chat_data = Dataset.from_list(sharegpt_chat_data)
print("Finished.")

# Tokenize Datasets
def encode(row):
    input_ids = []
    for line in row["input"]:
        input_ids.append((line[0], tokenizer.encode(line[1])))
    
    row["input_ids"] = input_ids
    return row

def filter_max_len_seqs(row):
    max_len = tokenizer.model_max_length

    inputs = []
    for line in row["input"]:
        if len(line[1]) < max_len:
            inputs.append((line[0], line[1]))
    
    row["input"] = inputs
    return row

for name, tokenizer in TOKENIZERS.items():
    print(f"Using {name} tokenizer.")
    path = os.path.abspath(os.path.join(os.getcwd(), name))

    if not os.path.exists(path):
        os.makedirs(path)

    filename = DATASETS_META["Aeala/ShareGPT_Vicuna_unfiltered"]["chat_dataset_file_name"]
    final_path = os.path.join(path, filename)

    print(f"Preparing {filename}...")
    dataset = []
    for row in tqdm(sharegpt_chat_data):
        filtered_row = filter_max_len_seqs(row)
        dataset.append(encode(filtered_row))


    saved_dataset = []
    for data in dataset:
        saved_dataset.append(data["input_ids"])

    print(f"Saving {filename}...")
    with open(final_path, 'wb') as f:
        pickle.dump(saved_dataset, f)
    print("Finished.")