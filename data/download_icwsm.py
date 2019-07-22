import requests
from zipfile import ZipFile
from pathlib import Path

"""
Script to download data used in "Graph Based Semi-supervised Learning with 
Convolutional Neural Networks to Classify Crisis Related Tweets" (Alam, 2018).
"""
QUEENSLAND_NEPAL_URL = "https://crisisnlp.qcri.org/data/acl_icwsm_2018/ACL_ICWSM_2018_datasets.zip"
DATA_PATH = Path(__file__).parent
QUEENSLAND_NEPAL_PATH = DATA_PATH / "ACL_ICWSM_2018_datasets.zip"

r = requests.get(QUEENSLAND_NEPAL_URL)

with QUEENSLAND_NEPAL_PATH.open("wb") as f:
    f.write(r.content)

with ZipFile(QUEENSLAND_NEPAL_PATH) as z:
    z.extractall(DATA_PATH)