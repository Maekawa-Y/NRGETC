import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.personalized_attention import Personalized_Attention


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.dense_layer = nn.Linear(config.user_embedding_dim, config.query_vector_dim)
        self.personalized_user_attention = Personalized_Attention(config.query_vector_dim, config.num_filters)

    def forward(self, user_vector, clicked_news_length, clicked_news_vector):
        
        # user preference query
        user_preference_query = F.dropout(F.relu(self.dense_layer(user_vector)),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        
        final_user_vector = self.personalized_user_attention(clicked_news_vector, user_preference_query, clicked_news_length)
        
        return final_user_vector