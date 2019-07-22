from typing import Dict, Iterable
import logging

from content_analyzer.data.dataset_readers import DATA_DIR

from pathlib import Path

from overrides import overrides

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('crisis_nlp')
class CrisisNLPDatasetReader(DatasetReader):
    """
    Reads tsv dataset provided by CrisisNLP along "Graph Based Semi-supervised
    Learning with Convolutional Neural Networks to Classify Crisis Related
    Tweets" (Alam, 2018).
    """
    ICWSM_DIR = "ACL_ICWSM_2018_datasets"
    FILEPATH_DICT = dict(
        neq_train="nepal/2015_Nepal_Earthquake_train.tsv",
        neq_dev="nepal/2015_Nepal_Earthquake_dev.tsv",
        neq_test="nepal/2015_Nepal_Earthquake_test.tsv",
        qfl_train="queensland/2013_Queensland_Floods_train.tsv",
        qfl_dev="queensland/2013_Queensland_Floods_dev.tsv",
        qfl_test="queensland/2013_Queensland_Floods_test.tsv",
    )

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        fp = DATA_DIR / self.ICWSM_DIR / self.FILEPATH_DICT[file_path]
        tsv_rows = fp.read_text(encoding="latin").split("\n")[1:-1] # skip header and last empty line
        for row in tsv_rows:
            yield self.text_to_instance(*row.split("\t"))

    @overrides
    def text_to_instance(self, id: str, text: str, label: str) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(label)
        return Instance(fields)
