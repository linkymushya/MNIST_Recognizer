# 手写数字识别模型 MNIST-Recognizer 

### 描述

本模型致力于通过简单的MLP全连接模型来对经典的MNIST数据集进行模型训练，并通过tkinter和PIL库设计GUI让用户进行数字绘制来验证模型的准确性

### 运行环境

- 所需要的库 : torch PIL numpy

- 建议python版本：3.10

### 运行方式

- 确保`data_preprocess.py`，`GUI.py`，`MNISTRecognizer.py`，`mnist_mlp.pth`与文件夹`data`在同一目录下
- 运行GUI.py文件，如果没有文件夹`data`，首先会运行预处理文件下载数据集，需要耐心等待，如果没有`mnist_mlp.pth`文件，会先对模型进行训练，训练完会保存模型数据，然后会出现绘图界面，此后便不再需要下载数据和训练模型，直接可以绘图
