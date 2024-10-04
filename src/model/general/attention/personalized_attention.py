import torch
import torch.nn as nn
import torch.nn.functional as F
from .masked_softmax import MaskedSoftmax


class Personalized_Attention(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(Personalized_Attention, self).__init__()
        self.linear = nn.Linear(query_vector_dim, candidate_vector_dim)
        self.masked_softmax = MaskedSoftmax()

    def forward(self, candidate_vector, user_preference_query, actual_lengths):
        
        temp = torch.tanh(self.linear(user_preference_query))
        
        if actual_lengths is None: # None mask
            print("Mask error")
            candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector),dim=1)
        else:
            candidate_weights = self.masked_softmax(torch.bmm(candidate_vector, temp.unsqueeze(2)).squeeze(2), actual_lengths)

        # batch_size, candidate_vector_dim
        target = torch.sum(candidate_weights.unsqueeze(2) * candidate_vector, dim=1)
        
        return target