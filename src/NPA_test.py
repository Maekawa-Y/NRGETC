import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from multiprocessing import Pool

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    
    return np.sum(rr_score) / np.sum(y_true)

def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}


# Create a news dictionary
news_parsed = pd.read_table(
    "../data/test/news_parsed.tsv",
    usecols=['id'] + config.dataset_attributes['news'],
    converters={
        attribute: literal_eval
        for attribute in set(config.dataset_attributes['news']) & set([
            'title'
        ])
    })
news_parsed["id"] = news_parsed["id"].astype(str)
news2dict = news_parsed.to_dict('index')
for key1 in news2dict.keys():
    for key2 in news2dict[key1].keys():
        if type(news2dict[key1][key2]) != str:
            news2dict[key1][key2] = torch.tensor(news2dict[key1][key2])
            
padding_all = {'title': [0] * config.num_words_title}
for key in padding_all.keys():
    padding_all[key] = torch.tensor(padding_all[key])

padding = {
    k: v
    for k, v in padding_all.items()
    if k in config.dataset_attributes['news']
}

user_to_int = pd.read_table('../data/train/user2int.tsv')
# Convert DataFrame to dictionary type
user_dictionary = dict(zip(user_to_int['user'], user_to_int['int']))

def search_user(key, default_value):
    return user_dictionary.get(key, default_value)

# Establish ID/index relationship
id_to_index = {entry['id']: index for index, entry in news2dict.items()}

class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(5),
                                       names=['impression_id', 'user', 'time', 'clicked_news', 'impressions'])
        self.behaviors.clicked_news = self.behaviors.clicked_news.fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors.iloc[idx]
        
        item["user"] = search_user(row.user, config.num_users)
        item["candidate_news"] = [
            news2dict[id_to_index.get(x.split('-')[0])] for x in row.impressions
        ]
        item["impressions"] = row.impressions
        
        item["clicked_news"] = [
            news2dict[id_to_index.get(x)]
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])    
        item["clicked_news_length"] = len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = item["clicked_news"] + [padding] * repeated_times
            
        return item

def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


@torch.no_grad()
def evaluate(model, directory, num_workers, max_count=sys.maxsize):

    behaviors_dataset = BehaviorsDataset(path.join(directory, 'behaviors.tsv'))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    count = 0
    tasks = []

    for minibatch in tqdm(behaviors_dataloader):
        count += 1
        if count == max_count:
            print("max_over")
            break        
        
        click_probability = model(minibatch["user"], minibatch["clicked_news_length"], minibatch["candidate_news"], minibatch["clicked_news"])
        
        y_true = [int(news[0].split('-')[1]) for news in minibatch['impressions']]
        y_pred = click_probability.squeeze().tolist()   
        
        tasks.append((y_true, y_pred))

    with Pool(processes=num_workers) as pool:
        results = pool.map(calculate_single_user_metric, tasks)

    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Evaluating model {model_name}')
    pretrained_word_embedding = torch.from_numpy(np.load('../data/train/pretrained_word_embedding.npy')).float()
    best_model_path = "./path/NPA/〇〇.pth" # Enter the path to save the model
    model = Model(config, pretrained_word_embedding).to(device)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    auc, mrr, ndcg5, ndcg10 = evaluate(model, '../data/test',config.num_workers)
    AUC = []
    MRR = []
    NDCG5 = []
    NDCG10 = []
    AUC.append(auc)
    MRR.append(mrr)
    NDCG5.append(ndcg5)
    NDCG10.append(ndcg10)
    print(f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}')
    print("-----------------------------")