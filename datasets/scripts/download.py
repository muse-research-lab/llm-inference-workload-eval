import os

from datasets import load_dataset

from utils import DATASETS_META, RAW_DATASETS_PATH

if not os.path.exists(RAW_DATASETS_PATH):
    os.makedirs(RAW_DATASETS_PATH)

for name, meta in DATASETS_META.items():
    version = meta["version"]
    split = "train"

    print(f"Downloading {name} dataset...")
    dataset = load_dataset(name, version=version, split=split)

    filename = meta["raw_file_name"]
    path = os.path.join(RAW_DATASETS_PATH, filename)
    
    print(f"Saving {name} dataset as {filename}...")
    dataset.to_json(path)

    print("Finished.")
