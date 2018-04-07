# TensorFlow实现一个简单的无Beam Search的NMT 

## 版本
Python3

TensorFlow 1.4+

## 数据
### 来源 
自制的文言文-现代文平行语料库，以字为单位(character-based)。
### 预处理运行方法
在`preprocess`目录下，运行
`python preprocess.py`。


### 用法
训练：`python main.py`

翻译：`python main.py -checkpoint models/*th_epoch_model_*.**.ckpt -translate`