from typing import Dict, Any

import argparse
import torch
import torch.nn as nn
import torchvision

from birdclef.models.util import GeM

H_SPEC = 128
W_SPEC = 313
EMBEDDING_SIZE = 1024

class ResNetBird(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        
        self.data_config = data_config
        self.input_dims = data_config["input_dims"]
        self.num_classes = data_config["output_dims"]
        
        embedding_size = self.args.get("embedding_size", EMBEDDING_SIZE)
        
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = GeM()
        # o_h, o_w = H_SPEC // 32, W_SPEC // 32 + 1
        self.embedding = nn.Linear(512, embedding_size)
        self.fc = nn.Linear(embedding_size, self.num_classes)
        
    def forward(self, x: torch.Tensor):
        x = self.resnet(x)
        x = self.pooling(x).flatten(1)
        x = self.embedding(x)
        x = self.fc(x)
        
        return x
    
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--embedding_size", type=int, default=EMBEDDING_SIZE)
    
    