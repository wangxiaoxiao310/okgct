# Image Classification 
- configs
  - WaiQ.yaml : 配置文件,需修改数据集,要训练的模型及checkpoint,log保存地址
- models：模型
- train.py : 训练文件
## 环境搭建
### pytorch 
1. pip install -r requirements.txt  # requirements.txt 位于 okgct下
2. 需搭建 pytorch1.8.1 gpu环境: 在官网 https://pytorch.org/ 选择如：
  `pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`
### mindspore
1. pip install -r requirements.txt
2. 需搭建 mindspore2.2.0 gpu环境: 在官网 https://www.mindspore.cn/install/ 选择如：
`conda install mindspore=2.2.0 -c mindspore -c conda-forge`
### 数据集
- 所需数据集位于 okgct/datasets下的 train_cifar10.zip， test_cifar10.zip， 需解压到某个位置，并在 configs/WaiQ.yaml中修改，见下
## 模型训练
1. 编辑 configs/WaiQ.yaml 文件选择模型：取消对应注释即可,需修改数据集及checkpoint,log保存地址
```python 
# 加载数据
dataset: 'cifar10' # 数据集名称
traindir: '/home/kgraph/workspace/data/train_cifar10' # 加载训练数据路径
testdir: '/home/kgraph/workspace//data/test_cifar10' # 加载测试数据路径

model: 'Resnet50_model' # 模型选择, 模型名称
#model: 'ShuffleNetV2_model' # 模型选择, 模型名称

checkpoint_save_dir: '/home/kgraph/workspace/data/torch/checkpoint/' # 模型保存路径
output_dir: '/home/kgraph/workspace/data/torch/logs/' # 输出保存结果的路径
```
2. 运行 train.py 文件
```bash
python train.py
```
3. 输出见：checkpoint 和 logs 文件夹