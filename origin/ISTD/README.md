# 环境搭建
## pytorch 
1. pip install -r requirements.txt  # requirements.txt 位于 okgct下
2. 需搭建 pytorch1.8.1 gpu环境: 在官网 https://pytorch.org/ 选择如：
  `pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116`
## mindspore
1. pip install -r requirements.txt
2. 需搭建 mindspore2.2.0 gpu环境: 在官网 https://www.mindspore.cn/install/ 选择如：
`conda install mindspore=2.2.0 -c mindspore -c conda-forge`
## 数据集
- 所需数据集位于 okgct/datasets下的 dataset.zip，** 注意需要解压到 ISTD 目录下**，及解压后dataset文件夹与 train.py为同级目录，否则需要参考 utils/parse_args.py文件指定--root参数
# 运行
默认UNet
python train.py  
python train.py --model PSPNet --lr 0.0001
# 结果
保存在 ISTD/checkpoints下
