import torch
from model.NAML.news_encoder import NewsEncoder
from model.NAML.user_encoder import UserEncoder
from model.general.click_predictor.dot_product import DotProductClickPredictor


class NAML(torch.nn.Module):

    def __init__(self, config, pretrained_word_embedding):
        super(NAML, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

    def forward(self, candidate_news, clicked_news, clicked_news_length):
    
        # batch_size, 1 + K, num_filters
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, num_filters
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, num_filters
        user_vector = self.user_encoder(clicked_news_vector, clicked_news_length)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        return click_probability

    def get_news_vector(self, news):
        # batch_size, num_filters
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector, clicked_news_length):
        # batch_size, num_filters
        return self.user_encoder(clicked_news_vector, clicked_news_length)

    def get_prediction(self, news_vector, user_vector):
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
