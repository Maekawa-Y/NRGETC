import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.additive import AdditiveAttention

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class TextEncoder(torch.nn.Module):
    def __init__(self, word_embedding, word_embedding_dim, num_filters,
                 window_size, query_vector_dim, dropout_probability):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.CNN = nn.Conv1d(word_embedding_dim, num_filters, window_size, padding=int((window_size - 1) / 2))
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(self.word_embedding(text),
                                p=self.dropout_probability,
                                training=self.training)
        # Permute dimensions for Conv1d input: (batch_size, word_embedding_dim, num_words_text)
        text_vector = text_vector.permute(0, 2, 1)
        # batch_size, num_filters, num_words_text
        convoluted_text_vector = self.CNN(text_vector)
        # batch_size, num_words_text, num_filters
        convoluted_text_vector = convoluted_text_vector.permute(0, 2, 1)
        # batch_size, num_filters, num_words_text
        activated_text_vector = F.dropout(F.relu(convoluted_text_vector),
                                          p=self.dropout_probability,
                                          training=self.training)
        
        # batch_size, num_filters
        actual_lengths = (text != 0).sum(dim=1).clone().detach()
        final_text_vector = self.additive_attention(activated_text_vector, actual_lengths)
        
        return final_text_vector

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0)
        assert len(config.dataset_attributes['news']) > 0
        text_encoders_candidates = ['title']
        self.text_encoders = nn.ModuleDict({
            name:
            TextEncoder(word_embedding, config.word_embedding_dim,
                        config.num_filters, config.window_size,
                        config.query_vector_dim, config.dropout_probability)
            for name in (set(config.dataset_attributes['news'])
                         & set(text_encoders_candidates))
        })
        
        if len(config.dataset_attributes['news']) > 1:
            self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                     config.num_filters)

    def forward(self, news):
        text_vectors = [
            encoder(news[name].to(device))
            for name, encoder in self.text_encoders.items()
        ]

        all_vectors = text_vectors

        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))
            
        return final_news_vector

    