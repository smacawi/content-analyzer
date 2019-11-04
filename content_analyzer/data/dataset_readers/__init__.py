from pathlib import Path
DATA_DIR = Path(__file__).parents[3] / 'data'

from content_analyzer.data.dataset_readers.icwsm import IcwsmCrisisNLPDatasetReader
from content_analyzer.data.dataset_readers.lrec import LrecCrisisNLPDatasetReader