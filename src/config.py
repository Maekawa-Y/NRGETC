import os

model_name = 'NRMS'
assert model_name in [
    'NRMS', 'NAML', 'LSTUR', 'NPA'
]


class BaseConfig():
    num_epochs = 50
    num_batches_show_loss = 100  # Number of batchs to show loss
    batch_size = 128
    learning_rate = 1e-4
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 30
    word_freq_threshold = 1
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 39064
    num_users = 1 + 46012
    word_embedding_dim = 300
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title'],
        "record": []
    }
    # For CNN
    num_filters = 400
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 400
    window_size = 3
    long_short_term_method = 'ini'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class NPAConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title'],
        "record": ['user']
    }
    # For CNN
    num_filters = 400 # 論文では400だった...
    window_size = 3
    user_embedding_dim = 50