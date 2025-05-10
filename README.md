# 🐶🐱 Cat-Dog Classifier v2

基于 PyTorch 实现的猫狗图像识别项目，在 v2 分支中使用了可选深度的 ResNet 网络结构，并支持命令行参数配置训练过程。适合深度学习初学者用于模型训练、推理及基础模型对比实验。

## 🌟 项目亮点

- ✅ 支持 `resnet18` / `resnet34` / `resnet50` 等不同深度的模型结构
- ✅ 模型训练与预测模块分离，结构清晰
- ✅ 命令行参数灵活配置，训练过程可控
- ✅ 数据加载模块化，支持数据增强
- ✅ 可保存 `.pth` 权重文件用于后续推理


## 📁 项目结构

```
dog-cat-classifiy/
├── dataset/               # 存放训练和验证图像
│   ├── train/
│   └── val/
├── src/
│   ├── model.py           # 模型构建（支持不同ResNet深度）
│   ├── data\_loader.py     # 数据加载与预处理
│   ├── train.py           # 模型训练脚本（支持命令行参数）
│   └── predict.py         # 单图预测脚本
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明文件（v2）
```


## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
````

### 2. 数据准备

请将猫狗图片分类放入如下结构：

```
dataset/
├── train/
│   ├── cats/
│   └── dogs/
└── val/
    ├── cats/
    └── dogs/
```

### 3. 模型训练

```bash
python src/train.py --depth resnet34 --epochs 10 --batch_size 64
```

参数说明：

* `--depth`: 使用的模型深度（默认：resnet34）
* `--epochs`: 训练轮数（默认：5）
* `--batch_size`: 每批图像数（默认：128）

训练完成后，模型将保存在项目根目录，如：

```
cat_dog_classifier_resnet34.pth
```

### 4. 图像预测

```bash
python src/predict.py path/to/image.jpg --depth resnet34 --weights cat_dog_classifier_resnet34.pth
```

你将看到类似输出：

```
Prediction: Cat (confidence: 98.3%)
```


### 5. Web启动
```bash
python app.py
```
## 🧠 模型说明

模型使用了 torchvision 提供的预训练 ResNet 网络，并将输出层修改为 2 类（猫 / 狗）分类。所有模型支持使用 GPU 加速。


## 🛠 开发与扩展建议

* 支持 Streamlit/Web 前端部署
* 添加混淆矩阵与分类报告分析
* 结合 YOLO 实现猫狗目标检测版本（master分支已实现）
* 引入 AutoAugment 等数据增强策略


## 📌 分支说明

* `master`：基础版本，结构简洁，适合入门
* `v2`：当前分支，支持多模型结构和命令行训练，适合实验与研究用途


## 📄 License

本项目仅供学习与研究使用，禁止用于商业用途。


欢迎 Star ⭐ 和 Fork 🍴！

```