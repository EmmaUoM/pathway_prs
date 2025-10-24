import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        # 独立的 LeakyReLU 实例，名称与每个全连接层对应
        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()
        self.leakyrelu3 = nn.LeakyReLU()

        # Dropout 层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.leakyrelu1(self.fc1(x)))
        x = self.dropout(self.leakyrelu2(self.fc2(x)))
        x = self.dropout(self.leakyrelu3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x)) 
        return x


def build_nn(input_dim):
    model = DNN(input_dim)
    return model
