import os
import shutil

import kagglehub

# downloads only the HI-Small files we use, the full Kaggle dataset has 6 variants (tens of GB)
# run from the project folder: python scripts/download_data.py

handle = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
raw_directory = "data/raw/"
os.makedirs(raw_directory, exist_ok=True)

for filename in ["HI-Small_Trans.csv", "HI-Small_accounts.csv", "HI-Small_Patterns.txt"]:
    path = kagglehub.dataset_download(handle, path=filename)
    shutil.copy(path, raw_directory + filename)
    print("got", filename, "->", raw_directory + filename)
