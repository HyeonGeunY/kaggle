from typing import Dict, Any

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
H_SPEC = 256
W_SPEC = 313




class ConvBlock(nn.Module):
    """
    input과 ouput의 이미지 크기가 동일한 CNN block (padding==1, stride==1, kernel_size==3)
    
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Con2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLu()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            torch.Tensor : (B, C, H, W) 
            
        """
        c = self.conv(x)
        r = self.relu(c)

        return r


class SimpleCNN(nn.Module):


    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:

        super().__init__()
        self.args = vars(args) if args is not None else {}

        self.input_dims = data_config["input_dims"]  # (C, H, W)
        self.num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)

        self.conv1 = ConvBlock(self.input_dims[0], conv_dim)
        self.conv2 = ConvBlock(conv_dim, conv_dim * 2)
        self.dropout = nn.Dropout(0.25)
        self.max_pool = nn.MaxPool2d(2)

        o_h, o_w = H_SPEC // 2, W_SPEC // 2

        fc_input_dim = int(o_h * o_w * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor size가 반드시 H_SPEC, W_SPEC과 일치해야 한다. (B, C, H, W)
        
        Returns:
            torch.Tensor: output tensor (B, n_classes)

        """
        _B, _C, H, W = x.shape
        assert H == H_SPEC
        assert W == W_SPEC

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        return parser
