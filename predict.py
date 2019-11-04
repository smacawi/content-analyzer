from allennlp.data.vocabulary import Vocabulary
from content_analyzer.models.rnn_classifier import RnnClassifier
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from content_analyzer.data.dataset_readers.twitter import TwitterNLPDatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
import torch
from allennlp.predictors import Predictor
from allennlp.predictors.text_classifier import TextClassifierPredictor
import overrides
from allennlp.common.util import JsonDict

indexer = PretrainedBertIndexer('bert-base-uncased')
wt = WordTokenizer()
tdr = TwitterNLPDatasetReader({"tokens": indexer}, wt)

GRU_args = {
    "bidirectional": True,
    "input_size": 768,
    "hidden_size": 768,
    "num_layers": 1,
}
print("vocab")
vocab = Vocabulary.from_files("out/flood_model/vocabulary")
print("embedder")
token_embedder = PretrainedBertEmbedder("bert-base-uncased")
text_embedder = BasicTextFieldEmbedder({"tokens": token_embedder}, allow_unmatched_keys = True)
print("encoder")
seq2vec = PytorchSeq2VecWrapper(torch.nn.GRU(batch_first=True, **GRU_args))
print("model")
model = RnnClassifier(vocab, text_embedder, seq2vec)

print("model state")
with open("out/flood_model/best.th", 'rb') as f:
    state_dict = torch.load(f)
    model.load_state_dict(state_dict)

predictor = TextClassifierPredictor(model, tdr)
prediction = predictor.predict("five people missing according to state police. if you have any information please contact us.")
print(prediction)