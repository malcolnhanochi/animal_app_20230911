# 必要なモジュールのインポート
from IPython.core.debugger import py3compat
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
import torch
import pandas as pd
import os

import matplotlib.pyplot as plt



from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import resnet18 

# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#　ネットワークの定義
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        #学習時に使ったのと同じ学習済みモデルを定義
        self.feature = resnet18(pretrained=True) 
        self.fc = nn.Linear(1000, 2)
        # self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        #学習時に使ったのと同じ順伝播
        # h = self.bn(x)
        h = self.feature(x)
        h = self.fc(h)
        return h