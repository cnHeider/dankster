import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms

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

use_gpu = torch.cuda.is_available()


def imshow(inp):
  inp = inp.numpy().transpose((1, 2, 0))
  mean = np.array([0.485, 0.456, 0.406])
  std = np.array([0.229, 0.224, 0.225])
  inp = std * inp + mean
  plt.imshow(inp)

inputs, classes = next(iter(dset_loaders['train']))

out = torchvision.utils.make_grid(inputs)

imshow(out)
plt.title([dset_classes[x] for x in classes])
plt.show()


def train_model(model, criterion, optim_scheduler, num_epochs=25):
  since = time.time()

  best_model = model
  best_acc = 0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        optimizer = optim_scheduler(model, epoch)

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      for data in dset_loaders[phase]:
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
          inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
          inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if phase == 'train':
          loss.backward()
          optimizer.step()

        # statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / dset_sizes[phase]
      epoch_acc = running_corrects / dset_sizes[phase]

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))

      # deep copy the model
      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model)

    print()

  time_elapsed = time.time() - since
  print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
  print('Best val Acc: {:4f}'.format(best_acc))
  return best_model


def visualize_model(model, num_images=5):
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

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

model.fc = nn.Linear(512, 100)

if use_gpu:
  model = model.cuda()

criterion = nn.CrossEntropyLoss()


def optim_scheduler_conv(model, epoch, init_lr=0.001, lr_decay_epoch=7):
  lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

  if epoch % lr_decay_epoch == 0:
    print('LR is set to {}'.format(lr))

  optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
  return optimizer

model = train_model(model, criterion, optim_scheduler_conv)

visualize_model(model)

torch.save(model, 'model')
# torch.save(model.state_dict(), 'model_params') # Recommended way to save model
