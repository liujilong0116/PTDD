# 线性层模型，为主题注意力决策模块

import torch
import torch.nn.functional as F

device = torch.device('cuda:0')

class MyLinear(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MyLinear, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden, bias=False)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)
        self.w1 = torch.ones([n_hidden, n_feature], requires_grad=True)
        self.b1 = torch.zeros(n_hidden, requires_grad=True)
        self.hidden_1.weight = torch.nn.parameter.Parameter(self.w1)
        self.hidden_1.bias = torch.nn.parameter.Parameter(self.b1)

        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = (torch.sigmoid(x) - torch.full(x.shape, 0.5).to(device)) * 2
        x = self.hidden_1(x)
        x = self.out(x)
        return torch.sigmoid(x)

