import torch
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, user_vector, clicked_news_length):
    
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(user_vector, clicked_news_length)
        # batch_size, word_embedding_dim
        final_user_vector = self.additive_attention(multihead_user_vector, clicked_news_length)
        
        return final_user_vector
