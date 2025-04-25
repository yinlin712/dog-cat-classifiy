from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
data_loader: 数据加载器，train过程中主要由loader对图片预处理
batch_size改成64 or 128会更快训练
"""

def get_data_loaders(train_dir, val_dir, batch_size=128):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # 统一尺寸，否则pytorch自带的loader报错
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
