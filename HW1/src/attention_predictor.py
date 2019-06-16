import torch
from base_predictor import BasePredictor
from modules import AttentionNet


class AttentionPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                 dropout_rate=0.2, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(AttentionPredictor, self).__init__(**kwargs)
        self.model = AttentionNet(embedding.size(1),
                                similarity=similarity) # include ExampleNet
        
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens']) # need to know
        loss = self.loss(logits, batch['labels'].float().to(self.device)) # this is batch loss, tensor.float() 轉float
        return logits, loss                                               # logit loss大接近1 小接近0
                                                                          #  logit 取1的維度
        # matrix.update(logits,)
        
    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
