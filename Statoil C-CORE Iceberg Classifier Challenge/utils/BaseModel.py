import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, pooling_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3)
        self.act = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=pooling_size, stride=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x:torch.tensor):
        x = self.conv(x)
        x = self.act(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x
        

class BasicFc(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
        

class BaseModel(nn.Module):

    def __init__(self, channels, pooling_sizes):
        super().__init__()
        self.conv1 = BasicConv(channels[0], channels[1], pooling_sizes[0])
        self.conv2 = BasicConv(channels[1], channels[2], pooling_sizes[1])
        self.conv3 = BasicConv(channels[2], channels[3], pooling_sizes[2])
        self.conv4 = BasicConv(channels[3], channels[4], pooling_sizes[3])
        self.fc1 = BasicFc(256, 512)
        self.fc2 = BasicFc(512, 256)
        
        self.fc3 = nn.Linear(256, 1)
        self.act3 = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1) # 2차원으로 만들기
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.act3(x)