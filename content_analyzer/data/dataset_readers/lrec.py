from typing import Dict, Iterable, List
import logging

from content_analyzer.data.dataset_readers import DATA_DIR

from pathlib import Path

from overrides import overrides
import pandas as pd

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers import Tokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('lrec')
class LrecCrisisNLPDatasetReader(DatasetReader):
    """
    Reads csv datasets provided by CrisisNLP along " Twitter as a Lifeline:
    Human-annotated Twitter Corpora for NLP of Crisis-related Messages."
    (Imran, 2016).

    In particular, this dataset reader allows reading from training sets for
    multiple events, to allow evaluating model generalizability across events
    and transfer learning effectiveness.
    """
    LREC_CF_DIR = "CrisisNLP_labeled_data_crowdflower"
    CF_FILEPATH_DICT = {
        '2013_pak_eq': "2013_Pakistan_eq/2013_pakistan_eq",
        '2014_cali_eq': "2014_California_Earthquake/2014_california_eq",
        '2014_chile_eq': "2014_Chile_Earthquake_en/2014_chile_eq_en",
        '2014_odile': "2014_Hurricane_Odile_Mexico_en/2014_hurricane_odile",
        '2014_india_floods': "2014_India_floods/2014_india_floods",
        '2014_pak_floods': "2014_Pakistan_floods/2014_pakistan_floods_cf_labels",
        '2014_hagupit': "2014_Philippines_Typhoon_Hagupit_en/2014_typhoon_hagupit_cf_labels",
        '2015_pam': "2015_Cyclone_Pam_en/2015_cyclone_pam_cf_labels",
        '2015_nepal_eq': "2015_Nepal_Earthquake_en/2015_nepal_eq_cf_labels",
    }

    def __init__(self,
                 train_events: List[str],
                 test_events: List[str],
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 label_col: str = 'choose_one_category',
                 text_col: str = 'tweet_text') -> None:
        super().__init__(lazy=lazy)
        self.train_events = train_events
        self.test_events = test_events
        self.label_col = label_col
        self.text_col = text_col
        self._tokenizer = tokenizer or Tokenizer(type="word")
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """Hacky method of reading from multiple files for test/train
        """
        df = pd.DataFrame()
        if file_path == 'train':
            events = self.train_events
        elif file_path == 'test':
            events = self.test_events
        for event in events:
            fp = DATA_DIR / self.LREC_CF_DIR / f"{self.CF_FILEPATH_DICT[event]}_{file_path}.csv"
            df = df.append(pd.read_csv(fp, encoding = "ISO-8859-1"))

        for i,row in df.iterrows():
            try:
                yield self.text_to_instance(row[self.text_col], row[self.label_col])
            except Exception as e:
                print("Invalid data:")
                print(row)
    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)
