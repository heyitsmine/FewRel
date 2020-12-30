import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn


class ProtoPlain(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = sentence_encoder.hidden_size
        self.drop = nn.Dropout()

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        # (B, 1, N, D) (B, total_Q, 1, D)
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        total_Q: Num of instances in the query set
        """
        support = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query)  # (B * total_Q, D)

        # # Dropout
        # support = self.drop(support)
        # query = self.drop(query)

        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size)  # (B, total_Q, D)

        # Prototypical Networks
        support = torch.mean(support, 2)  # Calculate prototype for each class, (B, N, D)
        logits = -self.__batch_dist__(support, query)  # (B, total_Q, N), 注意加了负号
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
