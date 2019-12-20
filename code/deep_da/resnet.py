import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

class Resnet(nn.Module):
    def __init__(self, num_class, bottleneck_width=4096, width=1024):
        super(Resnet, self).__init__()
        self.base_network = models.resnet50(pretrained=True)

        self.classifier_layer = nn.Sequential(nn.Linear(self.base_network.fc.out_features, num_class), nn.Softmax())

        for layer in self.classifier_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.fill_(0.0)

    def forward(self, source):
        source = self.base_network(source)

        source_clf = self.classifier_layer(source)

        return source_clf

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf