#!/usr/bin/env python
# coding: utf-8
import json, time, os, sys, logging
from datetime import datetime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sb                                     
import numpy as np


logger = logging.getLogger(__name__)


def load_data(train_dir, valid_dir, test_dir, batch_size):
    """ Loading data from directories 

    :train_dir: directory with training samples
    :valid_dir: directory with valid samples
    :test_dir: directory with testing samples
    :returns: {'train_transforms': train_transforms,
                'test_transforms': test_transforms,
                'valid_transforms': valid_transforms,
                'train_data':   train_data,
                'test_data':   train_data,
                'valid_data':   train_data,
                'trainloader':  trainloader,
                'testloader': testloader,
                'validloader':  validloader,
                }
    """ 
    train_transforms = transforms.Compose([                                         
        transforms.Resize((224,224)),
        transforms.RandomCrop(224),                                               
        transforms.RandomRotation(30),                                            
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])                                                                          
    test_transforms = transforms.Compose([                                         
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),                                                
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  
    valid_transforms = transforms.Compose([                                         
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),                                               
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data,
            batch_size=batch_size , shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,
            batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data,
            batch_size=batch_size)

    return {'train_transforms': train_transforms,
            'test_transforms': test_transforms,
            'valid_transforms': valid_transforms,
            'train_data':   train_data,
            'test_data':   train_data,
            'valid_data':   train_data,
            'trainloader':  trainloader,
            'testloader': testloader,
            'validloader':  validloader,
            }


def validation(model, validloader, criterion, device):
    """ Implement a function for the validation pass """
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def train(model, epochs, data, optimizer, criterion, device, logger):
    """ Train NN """
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        start_ts = datetime.now()
        model.train()
        for images, labels in data['trainloader']:
            steps += 1
            images, labels = images.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(
                            model, data['testloader'], criterion, device)
                            
                logger.info(f'Epoch: {e+1}/{epochs}.. ' + \
                    f'Training Loss: {running_loss/print_every:.3f}.. ' + \
                    f'Test Loss: {test_loss/len(data["testloader"]):.3f}.. '+\
                    f'Test Accuracy: {accuracy/len(data["testloader"]):.3f} '+\
                    f'Runtime: {datetime.now() - start_ts}')
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()
    return model

def check_dir(DIR):
    """ check for existance and make one"""
    if not os.path.exists(DIR):
        logger.info('There is no dir: %r I\'ll make one ', DIR)
        os.makedirs(DIR, mode=0o755)
        return True
    else:
        return False

def label_map(args):
    """ Returns Label Map """
    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    num_of_fw_classes = len(cat_to_name.keys())
    return num_of_fw_classes, cat_to_name

def create_model(args):
    """ Create Model from pretrained NN

    :args: CLI arguments
    :returns: model of NN

    """
    model = models.__dict__[args.arch](pretrained=True)
    return model

def load_model(args, checkpoint_file, num_of_fw_classes):
    ''' load checkpoint for args.arch ''' 
    ch_points = torch.load(checkpoint_file)
    model = models.__dict__[args.arch](pretrained=True)
    model = change_classifier(args, model, num_of_fw_classes) 
    model.load_state_dict(ch_points)
    return model

def change_classifier(args, model, num_of_fw_classes):
    """ Change classifier model to number of classes """
    # Freeze parameters so we don't backprop through them
    logger.info(f'Freeze parameter of the model.. ')
    for param in model.parameters():
        param.requires_grad = False
    # Change classifier 
    # new_in_features = model.classifier[0].in_features
    new_in_features = int()
    for seq in model.classifier:
        # print(f' {seq}, type -> {type(seq)}')
        if type(seq) == torch.nn.modules.linear.Linear:
            new_in_features = seq.in_features
            break

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(new_in_features, 4096)),
        ('relu1', nn.ReLU()),
        ('drp1', nn.Dropout(p=0.48)),
        ('fc2', nn.Linear(4096, args.hidden_units)),
        ('relu2', nn.ReLU()),
        ('drp2', nn.Dropout(p=0.52)),
        ('fc3', nn.Linear(args.hidden_units, num_of_fw_classes)),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    logger.info(f'Change model classifier.. ')
    model.classifier = classifier
    return model
 
