import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv1d(config.word_embedding_dim, config.num_filters, config.window_size, padding=int((config.window_size - 1) / 2))
        self.title_attention = AdditiveAttention(config.query_vector_dim, config.num_filters)

    def forward(self, news):
        
        # Part 3: calculate weighted_title_vector

        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(self.word_embedding(news['title'].to(device)),
                                 p=self.config.dropout_probability,
                                 training=self.training)
        # Permute dimensions for Conv1d input: (batch_size, word_embedding_dim, num_words_text)
        title_vector = title_vector.permute(0, 2, 1)
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(title_vector).permute(0, 2, 1)
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        
        title_actual_lengths = (news["title"] != 0).sum(dim=1).clone().detach()
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(activated_title_vector, title_actual_lengths)

        return weighted_title_vector
