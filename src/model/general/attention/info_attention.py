import torch
import torch.nn as nn
import torch.nn.functional as F

class additiveattention(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(additiveattention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector):
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        if temp.dim() <= 2:
            candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector),dim=0)
        else:
            candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector),dim=1)
        
        return candidate_weights