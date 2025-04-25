import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import os
from io import BytesIO
import base64

# 模型加载
model_path = "runs/detect/cat_dog_yolov817/weights/best.pt"
model = YOLO(model_path)

# 页面标题
st.title("🐾 YOLOv8 猫狗识别")

# 初始化会话状态保存识别历史
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("请上传一张猫或狗的图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原图
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="你上传的图片", use_container_width=True)

    # 保存临时图像
    temp_image_path = "temp_uploaded_image.jpg"
    image.save(temp_image_path)

    # 推理
    results = model(temp_image_path)
    res_image = results[0].plot()  # 带检测框图像 (np.ndarray)
    boxes = results[0].boxes

    # 提取预测信息
    predictions = []
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = "🐱 猫咪" if cls == 0 else "🐶 狗狗"
        predictions.append((label, conf))

    # 显示预测信息
    for label, conf in predictions:
        st.success(f"{label} - 置信度: {conf:.2f}")

    # 显示识别图像
    st.image(res_image, caption="识别结果", use_container_width=True)

    # 保存预测历史（图像+结果）
    buffer = BytesIO()
    Image.fromarray(res_image).save(buffer, format="PNG")
    st.session_state.history.append({
        "image": buffer.getvalue(),
        "predictions": predictions
    })

    # 下载按钮
    st.markdown("### 📥 下载识别结果")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="cat_dog_result.png">点击下载结果图片</a>'
    st.markdown(href, unsafe_allow_html=True)

# 显示识别历史
if st.session_state.history:
    st.markdown("### 🕘 识别历史")
    for idx, item in enumerate(reversed(st.session_state.history)):
        st.image(item["image"], caption=f"历史记录 #{len(st.session_state.history) - idx}", use_container_width=True)
        for label, conf in item["predictions"]:
            st.write(f"{label} - 置信度: {conf:.2f}")
