
# 🐱🐶 Cat-Dog Classifier (猫狗识别 AI)

一个基于 PyTorch 和 ResNet18 构建的猫狗图像识别项目，支持通过网页上传图片识别是否是猫还是狗。
最新版实现了PyTorch + YOLOv8的实现版。

<a href='https://github.com/yinlin712/dog-cat-classifiy/tree/v2'>点我切换到v2分支</a>

SYLU深度学习课程大作业，built by yinlin

**5月10日更新：doc目录下增加了Latex作业报告文档**

## 📦 项目结构

```
dog-cat-classifiy/
├── dataset/                   # 数据集（ResNet）
│   ├── train/                 # 训练集（按 cats/dogs 分类）
│   │   ├── dogs/
│   │   └── cats/
│   └── val/                   # 验证集（按 cats/dogs 分类）
│       ├── dogs/
│       └── cats/
├── doc/                       # Latex报告文档，建议使用Overleaf打开，使用Xelatex编译器
├── runs/                      # yolo训练输出
├── src/
│   ├── model.py              # ResNet 模型构建
│   ├── convert_to_yolo.py    # 数据集转为yolo格式
│   ├── data_loader.py        # 数据加载器
│   ├── train.py              # 模型训练脚本
│   └── predict.py            # 命令行预测脚本
├── yolo_dataset/             # yolo数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml     
├── app.py                    # Streamlit Web 页面
├── data.py                   # 数据集获取脚本
├── requirements.txt          # 所需 Python 依赖
├── yolo_app.py               # yolo Web页面
├── yolov8n.pt
├── cat_dog_classifier.pth
└── README.md                 # 项目说明
```

---

## 🚀 快速开始

本项目数据集，训练模型、`best.pt`等已经打包上传，如果只想运行项目的话配置好环境直接跳转**启动网页界面**即可。

### 1️⃣ 安装环境

建议使用 Conda 环境(Python≥3.9)：

```bash
conda create -n catdog python=3.9 -y
conda activate catdog
pip install -r requirements.txt
```

如果安装完 `requirements.txt`有报错，请手动安装包依赖：

```bash
pip install torch torchvision streamlit Pillow
```

---

### 2️⃣ 准备数据集

数据集来源于Kaggle[Kaggle猫狗数据集](https://www.kaggle.com/datasets/tongpython/cat-and-dog)，运行`data.py`下载并查看数据集下载目录，
注意，数据集下载完毕后目录有重复混乱，请将 Kaggle 猫狗数据集整理划分为如下结构并将`dataset`移动到项目根目录：

```
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

每个目录下放置格式为 `cat.4001.jpg`、`dog.4001.jpg` 的图片。（数据集已准备好）

注意`yolo_dataset`生成，运行：
```bash
python src/convert_to_yolo.py
```

---

### 3️⃣ 训练模型

```bash
python -m src.train dataset/train dataset/val
```

训练完成后将生成模型文件：

```
cat_dog_classifier.pth
```

---

### 4️⃣ 启动网页界面

```bash
streamlit run app.py
```

YOLO版
```bash
streamlit run yolo_app.py
```

打开浏览器访问 [http://localhost:8501](http://localhost:8501)

你可以上传一张猫或狗的照片，AI 会告诉你是哪种动物。

---

### 5️⃣ 命令行预测（可选）

```bash
python -m src.predict dataset/val/cats/cat.4001.jpg
```

---

## ⚙️ 使用技巧

- 支持 GPU 自动加速（使用 `torch.cuda.is_available()` 判断）。
- 图片会自动缩放为 `224x224`。
- 支持 `.jpg`, `.jpeg`, `.png` 格式。
- 使用 ResNet18 作为骨干网络，预训练自 ImageNet。

---

## 🧠 模型结构说明

使用 `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` 加载预训练权重，并修改最后的全连接层为：

```python
self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 2类：猫和狗
```

---

## 📝 License

本项目仅供学习交流，禁止用于商业用途。如需引用或二次开发请注明来源。
