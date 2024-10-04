import torch
import torch.nn as nn
import torch.nn.functional as F
from model.NPA.news_encoder import NewsEncoder
from model.NPA.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NPA(torch.nn.Module): 
    
    def __init__(self, config, pretrained_word_embedding):
        super(NPA, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        
        self.user_embedding = nn.Embedding(config.num_users+1, config.user_embedding_dim, padding_idx=0)

    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
        
        user_vector = self.user_embedding(user.to(device))
        
        candidate_news_vector = torch.stack([self.news_encoder(x, user_vector) for x in candidate_news], dim=1)

        clicked_news_vector = torch.stack([self.news_encoder(x, user_vector) for x in clicked_news], dim=1)
        
        user_vector = self.user_encoder(user_vector, clicked_news_length, clicked_news_vector)
        
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,user_vector)
        
        return click_probability