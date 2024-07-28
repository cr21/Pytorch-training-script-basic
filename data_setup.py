

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloader(train_path:str,
                      test_path:str,
                      valid_path:str,
                      test_transforms:transforms,
                      train_transforms:transforms,
                      num_of_workers:int,
                      batch_size:int):
  """
  create train, test and valid dataloader


  """
  train_dataset=datasets.ImageFolder(train_path, transform=train_transforms)
  test_dataset=datasets.ImageFolder(test_path, transform=test_transforms)
  valid_dataset=datasets.ImageFolder(valid_path, transform=test_transforms)

  class_names = train_dataset.classes

  train_dataloader=DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_of_workers,
                              pin_memory=True)
  test_dataloader=DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_of_workers,
                              pin_memory=True)
  valid_dataset=DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_of_workers,
                              pin_memory=True)
  return train_dataloader, test_dataloader, valid_dataset, class_names


