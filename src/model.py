import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

"""
ResNet18轻量模型
"""
def create_model(num_classes=2):

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
