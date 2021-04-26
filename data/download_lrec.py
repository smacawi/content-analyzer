import requests
from zipfile import ZipFile
from pathlib import Path
import pandas as pd


"""
Script to download data used in " Twitter as a Lifeline: Human-annotated Twitter
 Corpora for NLP of Crisis-related Messages" (Imran, 2016).
"""

crowdflower = True

if crowdflower:
    CRISIS_NLP_URL = "https://crisisnlp.qcri.org/data/lrec2016/labeled_cf/CrisisNLP_labeled_data_crowdflower.zip"
    CRISIS_NLP_DIR = "CrisisNLP_labeled_data_crowdflower"
    label_col = "choose_one_category"

else:
    CRISIS_NLP_URL = "https://crisisnlp.qcri.org/data/lrec2016/labeled_aidr/CrisisNLP_volunteers_labeled_data.zip"
    CRISIS_NLP_DIR = "CrisisNLP_volunteers_labeled_data"
    label_col = "label"

DATA_PATH = Path(__file__).parent
CRISIS_NLP_PATH = DATA_PATH / "LREC_2016_datasets.zip"

r = requests.get(CRISIS_NLP_URL)

with CRISIS_NLP_PATH.open("wb") as f:
    f.write(r.content)

with ZipFile(CRISIS_NLP_PATH) as z:
    z.extractall(DATA_PATH)

for zip_file in (DATA_PATH / CRISIS_NLP_DIR).glob("*/*.zip"):
    print(f"unzipping {zip_file}")
    with ZipFile(zip_file) as z:
        z.extractall(zip_file.parent)


def create_train_test_split(csv_path: Path, label_col, split: float = 0.8):
    df = pd.read_csv(csv_path, encoding = "ISO-8859-1").sample(frac=1).reset_index(drop=True)
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    train_path = csv_path.parent / f"{csv_path.stem}_train.csv"
    test_path = csv_path.parent / f"{csv_path.stem}_test.csv"
    for label in df[label_col].unique():
        label_df = df.loc[df[label_col] == label]
        train_df = train_df.append(label_df[:int(len(label_df)*split)])
        test_df = test_df.append(label_df[int(len(label_df)*split):])

    print(f"Train set:\n{train_df[label_col].value_counts()}")
    print(f"Test set:\n{test_df[label_col].value_counts()}")
    print("-"*80)
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)


split = 0.8
for csv_path in (DATA_PATH / CRISIS_NLP_DIR).glob("*/*.csv"):
    print(f"Splitting: {csv_path.parent}")
    create_train_test_split(csv_path, label_col, split)




