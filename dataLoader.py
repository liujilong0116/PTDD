import numpy as np
import torch.utils.data as Data
from tools import *
import torch
import os


delete_idx = ['451', '458', '480']

def loadTopicScoreForVote(batch_size, shuffle, root_path='./result/it_makes_me_good_bad_bert-base-uncased_with_question/', parameter=1.):
    class WozDataset(Data.Dataset):
        def __init__(self, data_list):
            true_lable_dict = {}   # 真实标签，{file_name:label}----{300:1}
            with open('./full_dataset.csv', 'r') as f:
                for line in f.readlines():
                    if not line.split(',')[0] == '':
                        true_lable_dict[line.split(',')[1]] = int(line.split(',')[2])
            x, y = [], []
            for data_name in data_list:
                if data_name not in delete_idx:
                    temp = readTopicScore(data_name+'.csv', root_path, parameter)
                    x.append(temp)
                    y.append(true_lable_dict[data_name])
            self.x = list(zip(x, y))

        def __getitem__(self, item):
            assert item < len(self.x)
            return self.x[item]

        def __len__(self):
            return len(self.x)

    train = WozDataset([i.split('.')[0] for i in os.listdir('./data/topic')])
    loader = Data.DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
    return loader

def loadTopicScoreForAttention(div, batch_size, shuffle, root_path='./data/that_sounds_good_bad_bert-base-uncased_with_question/', parameter=1.):
    class WozDataset(Data.Dataset):
        def __init__(self, data_list):
            true_lable_dict = {}   # 真实标签，{file_name:label}----{300:1}
            with open('./full_dataset.csv', 'r') as f:
                for line in f.readlines():
                    if not line.split(',')[0] == '':
                        true_lable_dict[line.split(',')[1]] = int(line.split(',')[2])
            x, y = [], []
            for data_name in data_list:
                if data_name not in delete_idx:
                    temp = readTopicScore(data_name+'.csv', root_path, parameter)
                    x.append(temp)
                    y.append(true_lable_dict[data_name])
            self.x = list(zip(x, y))

        def __getitem__(self, item):
            assert item < len(self.x)
            return self.x[item]

        def __len__(self):
            return len(self.x)

    data_div = np.load('./%s'%div, allow_pickle=True).item()
    train = WozDataset(data_div['train'])
    train_loader = Data.DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
    dev = WozDataset(data_div['dev'])
    dev_loader = Data.DataLoader(dataset=dev, batch_size=batch_size, shuffle=shuffle)
    test = WozDataset(data_div['test'])
    test_loader = Data.DataLoader(dataset=test, batch_size=batch_size, shuffle=shuffle)
    return train_loader, dev_loader, test_loader