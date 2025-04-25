import torch
import torch.optim as optim
from src.model import create_model
from src.data_loader import get_data_loaders
import torch.nn as nn
from tqdm import tqdm


"""
更改epoch，可以改训练的轮次
《泼墨漓江》真好听啊
"""
def train_model(train_dir, val_dir, epochs=3):
    train_loader, val_loader = get_data_loaders(train_dir, val_dir)

    # 使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("当前使用GPU训练和预测模型" if torch.cuda.is_available() else "当前使用CPU训练和预测模型")

    # 创建模型
    model = create_model()
    model.to(device)

    # 定义损失函数和优化器
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

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

    # 保存训练好的模型
    torch.save(model.state_dict(), 'cat_dog_classifier.pth')

if __name__ == '__main__':
    train_model(train_dir='dataset/train',val_dir='dataset/val')
