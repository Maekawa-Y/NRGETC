import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0)

        self.multihead_self_attention = MultiHeadSelfAttention(config.word_embedding_dim, config.num_attention_heads)
        self.additive_attention = AdditiveAttention(config.query_vector_dim, config.word_embedding_dim)

    def forward(self, news):
        
        news_title_actual_lengths = (news["title"] != 0).sum(dim=1).clone().detach()
        
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(self.word_embedding(news["title"].to(device)),
                                p=self.config.dropout_probability,
                                training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector, news_title_actual_lengths)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector, news_title_actual_lengths)
        
        return final_news_vector
