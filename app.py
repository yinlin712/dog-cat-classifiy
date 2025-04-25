import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model import create_model

# 模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model()
model.load_state_dict(torch.load("cat_dog_classifier.pth", map_location=device))
model.to(device)  # 把模型转移到 GPU（或 CPU）
model.eval()


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 页面
st.title("🐱🐶 猫狗识别 AI")
uploaded_file = st.file_uploader("上传一张猫或狗的图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="你上传的图片", use_container_width=True)


    # 推理
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        label = "🐱 猫咪" if predicted.item() == 0 else "🐶 狗狗"
        st.success(f"识别结果：{label}")
