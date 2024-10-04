import torch
from model.NRMS.news_encoder import NewsEncoder
from model.NRMS.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class NRMS(torch.nn.Module):
    
    def __init__(self, config, pretrained_word_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_length):
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = torch.stack([self.news_encoder(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack([self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector, clicked_news_length)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        
        return click_probability

    def get_news_vector(self, news):
        # batch_size, word_embedding_dim
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector, clicked_news_length):
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector, clicked_news_length)

    def get_prediction(self, news_vector, user_vector):
        # candidate_size
        return self.click_predictor(news_vector.unsqueeze(dim=0),user_vector.unsqueeze(dim=0)).squeeze(dim=0)
