# src/validate.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import create_model

def validate_model(val_dir, model_path='cat_dog_classifier.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'验证集准确率：{acc:.2f}%')

if __name__ == '__main__':
    import sys
    val_dir = sys.argv[1] if len(sys.argv) > 1 else 'dataset/val'
    validate_model(val_dir)
