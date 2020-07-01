# Deep Context Convolutional Neural Networks for Semantic Segmentation

说明：本仓库为非官方复现论文 Deep Context Convolutional Neural Networks for Semantic Segmentation

模型结构原理图：

<img src="C:\Users\wuyang\Desktop\Paddle\DCCNet_Paddle\src\Snipaste_2020-07-01_19-30-19.png" style="zoom:80%;" />

基本流程：

DCCNet先使用VGG-16作为backbone，进一步处理1/8，1/16，1/32的 feature map ，然后使用3个deconvolution layer扩大至相同大小size，得到最终prediction。

实验结果对比：

本文使用PASCAL VOC 2012数据集测试模型性能，所以我们也使用该数据集测试模型。

|     Method     | backbone | mIoU |
| :------------: | :------: | :--: |
| DCCNet（论文） |  VGG-16  | 71.4 |
|     DCCNet     |  VGG-16  |      |

