import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

"""
根据指定网络深度创建分类模型。
支持: resnet18, resnet34, resnet50
"""

def create_model(depth='resnet18', num_classes=2):
    if depth == 'resnet18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif depth == 'resnet34':
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    elif depth == 'resnet50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    else:
        raise ValueError("Unsupported depth. Use 'resnet18', 'resnet34', or 'resnet50'.")

    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
