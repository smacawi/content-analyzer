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
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('twitter')
class TwitterNLPDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False,
                 text_col: str = 'text') -> None:
        super().__init__(lazy=lazy)
        self.text_col = text_col
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """Hacky method of reading from multiple files for test/train
        """
        df = pd.read_csv(file_path)
        for i,row in df.iterrows():
            try:
                yield self.text_to_instance(row[self.text_col])
            except Exception as e:
                print("Invalid data:")
                print(row)
                raise e
    @overrides
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        return Instance(fields)
