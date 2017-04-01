import os
import cv2

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from time import sleep

from my_utils.camera import get_frame
from my_utils.show_image import show_image

use_gpu = torch.cuda.is_available()

model = torch.load('model')

if use_gpu:
  model = model.cuda()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/heider/Pictures/Webcam'
datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'val']}
dataset_loaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
dataset_classes = datasets['train'].classes


def get_prediction():
  data =  list(dataset_loaders['val'])[np.random.randint(0, 10)]
  inputs, labels = data
  if use_gpu:
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
  else:
    inputs, labels = Variable(inputs), Variable(labels)

  outputs = model(inputs)
  _, preds = torch.max(outputs.data, 1)

  action = dataset_classes[labels.data[np.argmax(preds)]]

  return action

if __name__ == '__main__':
  while True:
    frame = get_frame()
    action = get_prediction()
    print(action)
    if(action == 'dab'):
      show_image('/home/heider/Pictures/Webcam/val/dab/2017-03-31-231309_7.jpg')
    #sleep(2)
