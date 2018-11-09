
-----------

![cnn](https://github.com/gopala-kr/CNNs/blob/master/resources/img/cnn.PNG)

----------------

#### CNN Architectures

  - [LeNet(1998)] [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)]
  - [LeNet-5 (2010)] [[paper](http://yann.lecun.com/exdb/lenet/)]
  - [AlexNet (2012)] [[paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)]
  - [ZFNet(2013)] [[paper](https://arxiv.org/pdf/1311.2901.pdf)]
  - [VGGNet (2014)] [[paper](https://arxiv.org/pdf/1409.1556.pdf)]
  - [GoogleNet/Inception(2014)] [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)]
  - [FCN(2014)] [[paper](https://arxiv.org/pdf/1411.4038.pdf)]
  - [RCNN(2014)] [[paper](https://arxiv.org/abs/1311.2524)]
  - [Deeply-supervised networks(2014)] [[paper](https://arxiv.org/pdf/1409.5185.pdf)]
  - [ResNet(2015)] [[paper](https://arxiv.org/pdf/1512.03385.pdf)]
  - [Ladder network(2015)] [[paper](https://arxiv.org/pdf/1507.02672.pdf)]
  - [YOLO(2015)] [[Paper](https://arxiv.org/pdf/1506.02640.pdf)]
  - [FractalNet (2016)] [[paper](https://arxiv.org/pdf/1605.07648.pdf)]
  - [PolyNet/Inception-Residual(2016)] [[paper](https://arxiv.org/pdf/1611.05725.pdf)]
  - [DenseNet(2016)] [[paper](https://arxiv.org/pdf/1608.06993.pdf)] [[code](https://github.com/liuzhuang13/DenseNet)] 
  - [SegNet(2016)] [[paper](https://arxiv.org/pdf/1511.00561.pdf)]
  - [fast region based CNN(2016)] [[paper](https://arxiv.org/pdf/1506.01497.pdf)]
  - [Look up based CNN(2016)] [[paper](https://arxiv.org/pdf/1611.06473.pdf)]
  - [Deep network with stochastic depth(2016)] [[paper](https://arxiv.org/pdf/1603.09382.pdf)]
  - [SqueezeNet(2017)] [[Paper](https://arxiv.org/pdf/1709.01507.pdf)] [[code](https://github.com/hujie-frank/SENet)]
  - [ResNeXt(2016)] [[paper](https://arxiv.org/pdf/1611.05431.pdf)]
  - [CapsNet(2017)] [[paper](https://arxiv.org/ftp/arxiv/papers/1805/1805.11195.pdf)]
  - [MobileNets(2017)] [[paper](https://arxiv.org/pdf/1704.04861.pdf)]
  - [Xception(2017)] [[paper](https://arxiv.org/abs/1610.02357)]
  - [IRCNN(2017)][[paper](https://arxiv.org/pdf/1704.03264.pdf)]
  - [ViP CNN(2017)] [[paper](https://arxiv.org/pdf/1702.07191.pdf)]


--------------

### Applications

   - [Image Classification](#cnn-architectures)   
   - [Object Recognition]
   - [Object Tracking]
   - [Object Localisation]
   - [Object Detection](#object-detection) 
   - [Semantic Segmentation](#semantic-segmentation)
   - [Image Captioning]
   - [Biomedical Imaging]
     - [MDNet]
     - [U-Net]
     - [R2U-Net]
   - [Remote Sensing]
   - [Video Analysis]
   - [Motion Detection]
   - [Human Pose Estimation]
   - [3D Vision]
   - [Face Recognition] 
  
------------
#### Object Detection

<p align="center">
  <img width="1000" src="https://github.com/hoya012/deep_learning_object_detection/blob/master/assets/deep_learning_object_detection_history.PNG" "Example of anomaly detection.">
</p>


- [R-CNN]
- [Fast R-CNN]
- [Faster R-CNN]
- [Light-Head R-CNN]
- [Cascade R-CNN]
- [SPP-Net]
- [YOLO]
- [YOLOv2]
- [YOLOv3]
- [YOLT]
- [SSD]
- [DSSD]
- [FSSD]
- [ESSD]
- [MDSSD]
- [Pelee]
- [Fire SSD]
- [R-FCN]
- [FPN]
- [DSOD]
- [RetinaNet]
- [MegNet]
- [RefineNet]
- [DetNet]
- [SSOD]
- [CornerNet]
- [3D Object Detection]
- [ZSD（Zero-Shot Object Detection）]
- [OSD（One-Shot object Detection）]
- [Weakly Supervised Object Detection]
- [Softer-NMS]
- [2018]
- [Other]

--------------

#### Semantic Segmentation

- U-Net [[arxiv](https://arxiv.org/pdf/1505.04597.pdf)][[Pytorch](https://github.com/tangzhenyu/SemanticSegmentation_DL/tree/master/U-net)]
- SegNet [[arxiv](https://arxiv.org/pdf/1511.00561.pdf)][[Caffe](https://github.com/alexgkendall/caffe-segnet)]
- DeepLab [[arxiv](https://arxiv.org/pdf/1606.00915.pdf)][[Caffe](https://bitbucket.org/deeplab/deeplab-public/)]
- FCN [[arxiv](https://arxiv.org/pdf/1605.06211.pdf)][[tensorflow](https://github.com/tangzhenyu/SemanticSegmentation_DL/tree/master/FCN)]
- ENet [[arxiv](https://arxiv.org/pdf/1606.02147.pdf)][[Caffe](https://github.com/TimoSaemann/ENet)]
- LinkNet [[arxiv](https://arxiv.org/pdf/1707.03718.pdf)][[Torch](https://github.com/e-lab/LinkNet)]
- DenseNet [[arxiv](https://arxiv.org/pdf/1608.06993.pdf)[]
- Tiramisu [[arxiv](https://arxiv.org/pdf/1611.09326.pdf)]
- DilatedNet [[arxiv](https://arxiv.org/pdf/1511.07122.pdf)]
- PixelNet [[arxiv](https://arxiv.org/pdf/1609.06694.pdf)][[Caffe](https://github.com/aayushbansal/PixelNet)]
- ICNet [[arxiv](https://arxiv.org/pdf/1704.08545.pdf)][[Caffe](https://github.com/hszhao/ICNet )]
- ERFNet [[arxiv](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)][[Torch](https://github.com/Eromera/erfnet )]
- RefineNet [[arxiv](https://arxiv.org/pdf/1611.06612.pdf)][[tensorflow](https://github.com/tangzhenyu/SemanticSegmentation_DL/tree/master/RefineNet)]
- PSPNet [[arxiv](https://arxiv.org/pdf/1612.01105.pdf),[pspnet](https://hszhao.github.io/projects/pspnet/)][[Caffe](https://github.com/hszhao/PSPNet )]
- Dilated convolution [[arxiv](https://arxiv.org/pdf/1511.07122.pdf)][[Caffe](https://github.com/fyu/dilation )]
- DeconvNet [[arxiv](https://arxiv.org/pdf/1505.04366.pdf)][[Caffe](http://cvlab.postech.ac.kr/research/deconvnet/ )]
- FRRN [[arxiv](https://arxiv.org/pdf/1611.08323.pdf)][[Lasagne](https://github.com/TobyPDE/FRRN )]
- GCN [[arxiv](https://arxiv.org/pdf/1703.02719.pdf)][[PyTorch](https://github.com/ZijunDeng/pytorch-semantic-segmentation )]
- LRR [[arxiv](https://arxiv.org/pdf/1605.02264.pdf)][[Matconvnet](https://github.com/golnazghiasi/LRR )]
- DUC, HDC [[arxiv](https://arxiv.org/pdf/1702.08502.pdf)][[PyTorch](https://github.com/ZijunDeng/pytorch-semantic-segmentation )]
- MultiNet [[arxiv](https://arxiv.org/pdf/1612.07695.pdf)] [[tensorflow1](https://github.com/MarvinTeichmann/MultiNet)[tensorflow2](https://github.com/MarvinTeichmann/KittiSeg)]
- Segaware [[arxiv](https://arxiv.org/pdf/1708.04607.pdf)][[Caffe](https://github.com/aharley/segaware )]
- Semantic Segmentation using Adversarial Networks [[arxiv](https://arxiv.org/pdf/1611.08408.pdf)] [[Chainer](https://github.com/oyam/Semantic-Segmentation-using-Adversarial-Networks )]
- In-Place Activated BatchNorm:obtain #1 positions [[arxiv](https://arxiv.org/abs/1712.02616)] [[Pytorch](https://github.com/mapillary/inplace_abn)]


--------------

### References

- [neural-network-papers](https://github.com/robertsdionne/neural-network-papers)
- [ref-implementations](https://github.com/gopala-kr/CNNs/blob/master/ref-implementations.md)
- [cnn](https://github.com/gopala-kr/CNNs/blob/master/cnn.md)
- [image-video-classification](https://github.com/gopala-kr/CNNs/blob/master/image-video-classification.md)
- [object-detection](https://github.com/gopala-kr/CNNs/blob/master/object-detection.md)
- [object-tracking-and-recognition](https://github.com/gopala-kr/CNNs/blob/master/object-tracking-and-recognition.md)
- [semantic-segmentation](https://github.com/gopala-kr/CNNs/blob/master/semantic-segmentation.md)
- [image-generation](https://github.com/gopala-kr/CNNs/blob/master/image-generation.md)
- [human-pose-estimation](https://github.com/gopala-kr/CNNs/blob/master/human-pose-estimation.md)
- [low-level-vision](https://github.com/gopala-kr/CNNs/blob/master/low-level-vision.md)
- [vision-and-nlp](https://github.com/gopala-kr/CNNs/blob/master/vision-and-nlp.md)
- [other](https://github.com/gopala-kr/CNNs/blob/master/other.md)
- [face-recognition](https://github.com/gopala-kr/CNNs/blob/master/face-recognition.md)
- [cv-datasets](https://github.com/gopala-kr/ConvNets/blob/master/cv-datasets.md)

------------

- [What I learned from competing against a ConvNet on ImageNet](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)
- [neural-network-architectures](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)
- [Real-time Object Detection with YOLO, YOLOv2 and now YOLOv3](https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

-------------

_**Maintainer**_

Gopala KR / @gopala-kr


--------------
