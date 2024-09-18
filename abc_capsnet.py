import torch
import torch.nn as nn
from feature_extraction import VGG18Extractor
from attention import AttentionLayer
from capsule_network import CapsuleNetwork1, CapsuleNetwork2

class ABC_CapsNet(nn.Module):
    def __init__(self):
        super(ABC_CapsNet, self).__init__()
        self.feature_extractor = VGG18Extractor()
        self.attention = AttentionLayer(feature_size=512, hidden_size=128)
        self.cn1 = CapsuleNetwork1()
        self.cn2 = CapsuleNetwork2()

    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = self.attention(features)
        capsule_output_1 = self.cn1(attended_features)
        output = self.cn2(capsule_output_1)
        return output
