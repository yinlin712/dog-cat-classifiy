import sys
import torch
from PIL import Image
from model import create_model
from torchvision import transforms

def predict_image(image_path, model_path='cat_dog_classifier_resnet34.pth', depth='resnet34'):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(depth=depth)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载图片并预测
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    class_names = ['cats', 'dogs']
    predicted_class = class_names[predicted.item()]
    print(f"图像: {image_path} → 预测分类: {predicted_class}")


if __name__ == '__main__':
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'cat_dog_classifier_resnet34.pth'

    # 从模型文件名自动推断 depth（如 resnet34）
    if 'resnet' in model_path:
        depth = model_path.split('_')[-1].split('.')[0]  # 'resnet34'
    else:
        depth = 'resnet18'  # 默认值

    predict_image(image_path, model_path, depth)



