import torch
import kagglehub

# 输出为True代表使用了GPU
# print(torch.cuda.is_available())



# 下载Kaggle数据集，数据集网址 https://www.kaggle.com/datasets/tongpython/cat-and-dog
path = kagglehub.dataset_download("tongpython/cat-and-dog")

print("Path to dataset files:", path)