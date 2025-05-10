import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import create_model
from data_loader import get_data_loaders
from tqdm import tqdm

def train_model(train_dir, val_dir, epochs=3, batch_size=64, depth='resnet18'):
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(depth=depth)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{depth}] Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), f'cat_dog_classifier_{depth}.pth')


"""
可以修改 train_model 中的 depth、batch_size 和 epochs 参数，决定默认训练的模型类型和训练策略；
或者在命令行中动态指定这些参数：
例如：
    python src/train.py --depth resnet50 --epochs 10 --batch_size 64
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a cat-dog classifier using a ResNet model.')
    parser.add_argument('--train_dir', type=str, default='dataset/train', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='dataset/val', help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--depth', type=str, default='resnet34', help='ResNet depth (e.g., resnet18, resnet34, resnet50)')

    args = parser.parse_args()

    train_model(train_dir=args.train_dir,
                val_dir=args.val_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                depth=args.depth)
