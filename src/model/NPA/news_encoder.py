import torch
import torch.nn as nn
import torch.nn.functional as F
from model.general.attention.personalized_attention import Personalized_Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding.from_pretrained(pretrained_word_embedding, freeze=False, padding_idx=0)
        self.title_CNN = nn.Conv1d(config.word_embedding_dim, config.num_filters, config.window_size, padding=int((config.window_size - 1) / 2))
        self.dense_layer = nn.Linear(config.user_embedding_dim, config.query_vector_dim)
        self.personalized_title_attention = Personalized_Attention(config.query_vector_dim, config.num_filters)

    def forward(self, news, user_vector):

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
        
        
        # user preference query
        user_preference_query = F.dropout(F.relu(self.dense_layer(user_vector)),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        
        
        
        title_actual_lengths = (news["title"] != 0).sum(dim=1).clone().detach()
        # batch_size, num_filters
        final_title_vector = self.personalized_title_attention(activated_title_vector, user_preference_query, title_actual_lengths)

        return final_title_vector