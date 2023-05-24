# README

The pytorch code of some common neural network models, running on MNIST dataet.

## Requirements

Python >=3.8

```bash
pip install -r requirements.txt
```

## Dataset

MNIST

- num of train data : num of valid data = 9 : 1

## Models

- LeNet: [《Gradient-Based Learning Applied to Document Recognition》](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- AlexNet: [《ImageNet classification with deep convolutional neural networks 》](https://dl.acm.org/doi/pdf/10.1145/3065386)
- ResNet: [《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385.pdf)
- DenseNet: [《Densely Connected Convolutional Networks》](https://arxiv.org/pdf/1608.06993.pdf)
- MobileNet：[《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》](https://arxiv.org/abs/1704.04861)

## Instructions

Run inference：

```bash
python inference.py \
--model AlexNet \
--lr 0.005 \
--dropout 0.5 \
--batchsize 64 \
--num_epochs 10
```

The output will be written to `log\$model_name.txt`

## Add new model or dataset

- **Add new model** : Define new model class in model.py
- **Add new dataset** : Define new Dataset and Dataloader class in utils.py 
