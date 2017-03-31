import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms

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

data_dir = '/home/heider/Github/dankster/data'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes


def imshow(inp):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  plt.imshow(inp)


def visualize_model(model, num_images=10):
  for i, data in enumerate(dset_loaders['val']):
    inputs, labels = data
    if use_gpu:
      inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
      inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    plt.figure()
    imshow(inputs.cpu().data[0])
    plt.title('pred: {}'.format(dset_classes[labels.data[0]]))
    plt.show()

    if i == num_images - 1:
      break

visualize_model(model)
