#CCWH: Convolutional  Attention Module

<p float="left">
  <img src ="figures/CCWH.png"  width="1000"/>
</p>

Authors - wangjunjie <sup>1†</sup>, liaoxiangyun <sup>1,2†</sup>



† - Denotes Equal Contribution

*Abstract - In recent years, the channel and spatial attention mechanism has shown great potential in improving the performance of deep convolution neural networks on different problems. However, attention nowadays is mostly global and coarse-grained, and has a high complexity cost, which brings a contradiction between performance and complexity to the use of attention mechanism. In this paper, Through the fine-grained and local perspective, and considering the balance with the complexity of the model,, we propose a stacked structure(CCWH) to capture the attention mechanism of information interaction between different dimensions. For an intermediate feature map, stacking attention will establish spatial connections by focusing on different dimensions and using residuals, and it could encoder the information between the channel and the spatial with negligible computational overhead. At the same time, we also focus on the local mutual information between channels. The stacking attention is small and lightweight, which can be plug-and-played into various popular networks. While improving the performance of the network,it can also reduce many parameters. We verify the effectiveness of the proposed method in classical visual tasks, including image classification based on CAIREF10, CAIREF00 and ImageNet data sets, and target detection based on MSCOCO and PascalVoc2007 data sets. In addition, we demonstrate the effectiveness of our proposed model through ablation experiments and visual interpretation of traditional methods. A series of experimental results verify our intuition that it is important to capture fine-grained and local information dependencies when introducing attention mechanism into feature map.*

## Training From Scratch
However, this repository includes all the code required to recreate the experiments mentioned in the paper. This sections provides the instructions required to run these experiments. Imagenet training code is based on the official PyTorch example.

To train a model on ImageNet, run `train_imagenet.py` with the desired model architecture and the path to the ImageNet dataset:

### Simple Training

```bash
python train_imagenet.py -a resnet18 [imagenet-folder]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

Note, however, that we do not provide model defintions for AlexNet, VGG, etc. Only the ResNet family and MobileNetV2 are officially supported.
