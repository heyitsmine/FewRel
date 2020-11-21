import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, feature_dim, K, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.K = K
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(K))
        
    def forward(self, x):
        B, N, K, feature_dim = x.shape

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, K)
        # eij: [B*N, K]

        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        # a: [B*N, K]

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        # a: [B*N, K]
        # x: [B, N, K, D]
        weighted_input = x * a.view(B, N, K, 1)
        return torch.sum(weighted_input, 2)


class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dropout=0.5, use_attention=False, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(p=dropout)
        self.dot = dot
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = Attention(768, 5)

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        total_Q: Num of instances in the query set
        """
        support_emb = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(query)  # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)
        # Prototypical Networks 
        # Ignore NA policy
        if self.use_attention:
            support = self.attention(support)
        else:
            support = torch.mean(support, 2)  # Calculate prototype for each class, (B, N, D)
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N), 负数
        minn, _ = logits.min(-1)  # (B, total_Q)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        return logits, pred
