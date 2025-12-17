import torch.nn as nn
import torch.nn.functional as F


class EncoderMiniBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32):
        super(EncoderMiniBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        return x