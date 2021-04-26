from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register('rnn_classifier')
class RnnClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 dropout: float = 0.,
                 label_namespace: str = 'labels',
                 initializer: InitializerApplicator = InitializerApplicator()
                 ) -> None:
        super().__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()
        self._dropout = nn.Dropout(dropout)
        self._num_labels = vocab.get_vocab_size(namespace=label_namespace)
        self._classification_layer = nn.Linear(self._classifier_input_dim, self._num_labels)
        self._loss = nn.CrossEntropyLoss()
        self.itol = vocab.get_index_to_token_vocabulary(namespace=label_namespace) # idx to label
        self._prf_metrics = {l: F1Measure(i) for i,l in self.itol.items()}
        self._accuracy = CategoricalAccuracy()
        self._label_counts = {l: 0 for i,l in self.itol.items()}
        self.oov_id = vocab.get_token_index(vocab._oov_token)
        self.oov = {}

        initializer(self)

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:

        embedded_text = self._text_field_embedder(tokens)

        mask = get_text_field_mask(tokens).float()
        encoded_text = self._dropout(self._seq2vec_encoder(embedded_text, mask=mask))

        logits = self._classification_layer(encoded_text)
        probs = F.softmax(logits, dim=1)

        output_dict = {'logits': logits, 'probs': probs}

        oov = {k: int((t==self.oov_id).sum()) for k,t in tokens.items()}
        for k,v in oov.items():
            if k not in self.oov:
                self.oov[k] = 0
            self.oov[k] += v

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            for metric in self._prf_metrics.values():
                metric(logits, label.squeeze(-1))
            for i,l in self.itol.items():
                self._label_counts[l] += (label==i).sum()
            self._accuracy(logits, label.squeeze(-1))
            output_dict['loss'] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {f"oov_{k}": v for k,v in self.oov.items()}
        # precision/recall/f1
        metrics.update({f"{metric_name}_P": metric.get_metric()[0] for metric_name, metric in self._prf_metrics.items()})
        metrics.update({f"{metric_name}_R": metric.get_metric()[1] for metric_name, metric in self._prf_metrics.items()})
        metrics.update({f"{metric_name}_F1": metric.get_metric(reset)[2] for metric_name, metric in self._prf_metrics.items()})
        metrics.update({"accuracy": self._accuracy.get_metric(reset)})
        metrics.update({f"{label_name}_count": int(c) for label_name, c in self._label_counts.items()})
        if reset:
            self.oov = {}
            self._label_counts = {l: 0 for i, l in self.itol.items()}

        return metrics