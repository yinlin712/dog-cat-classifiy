import sys

import torch
from PIL import Image
from src.model import create_model
from torchvision import transforms


def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load('cat_dog_classifier.pth'))
    model.to(device)

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    class_names = ['cats', 'dogs']
    predicted_class = class_names[predicted.item()]
    print(f"预测分类: {predicted_class}")

if __name__ == '__main__':
    predict_image(sys.argv[1])
