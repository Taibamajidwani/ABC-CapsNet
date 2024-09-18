import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(feature_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention_weights = self.softmax(self.attention(x))
        attended_output = attention_weights * x
        return attended_output
