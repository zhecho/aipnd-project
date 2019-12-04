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
from PIL import Image


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
            'test_data':   test_data,
            'valid_data':   valid_data,
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

def load_model(checkpoint_file):
    ''' load from checkpoint file saved like this:
        checkpoint_dict = {
                'weights': model.state_dict(),
                'model_name': args.arch,
                'hidden_units': args.hidden_units,
                'output_units': num_of_fw_classes,
                'class_to_idx': model.class_to_idx }
        torch.save(checkpoint_dict, saved_pth_file)
    ''' 
    checkpoint_dict = torch.load(checkpoint_file)
    model = getattr(models, checkpoint_dict['model_name'])(pretrained=True)
    model = change_classifier(model, checkpoint_dict) 
    model.load_state_dict(checkpoint_dict['weights'])
    return model

def change_classifier(model, checkpoint_dict):
    """ Change classifier model to number of classes
        Supported nets starts with 'vgg','densenet','resnet','alexnet'
    """
    logger.info(f'Freeze parameter of the model.. ')
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier 
    # new_in_features = model.classifier[0].in_features
    new_in_features = int()
    if checkpoint_dict['model_name'].startswith(('vgg','alexnet')):
        for seq in model.classifier:
            # print(f' {seq}, type -> {type(seq)}')
            if type(seq) == torch.nn.modules.linear.Linear:
                new_in_features = seq.in_features
                break
    elif checkpoint_dict['model_name'].startswith('densenet'):
        new_in_features = model.classifier.in_features
    elif checkpoint_dict['model_name'].startswith('resnet'):
        new_in_features = model.fc.in_features

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(new_in_features, 4096)),
        ('relu1', nn.ReLU()),
        ('drp1', nn.Dropout(p=0.48)),
        ('fc2', nn.Linear(4096, checkpoint_dict['hidden_units'])),
        ('relu2', nn.ReLU()),
        ('drp2', nn.Dropout(p=0.52)),
        ('fc3', nn.Linear(checkpoint_dict['hidden_units'],
            checkpoint_dict['output_units'])),
        ('output', nn.LogSoftmax(dim=1))
        ]))
    logger.info(f'Change model classifier.. ')
    
    if checkpoint_dict['model_name'].startswith(('vgg','densenet','alexnet')):
        model.classifier = classifier
    elif checkpoint_dict['model_name'].startswith('resnet'):
        model.fc = classifier
    return model
 

def process_image(image):
    ''' Scales, crops, and t a PIL image for a PyTorch model, returns an Numpty
    array '''

    img = Image.open(image)
    '''Get the dimensions of the image'''
    width, height = img.size
    
    '''Resize by keeping the aspect ratio, but changing the dimension
    so the shortest size is 256px'''
    img = img.resize(
            (256, int(256*(height/width))) if width < height else
            (int(256*(width/height)), 256)
            )
    
    '''Get the dimensions of the new image size'''
    width, height = img.size
    
    '''Set the coordinates to do a center crop of 224 x 224'''
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))

    # img = img.resize((256,256))
    # img = img.crop((16,16,240,240))
    
    # to numpy array
    img = np.asarray(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = (img - mean)/std
    #  
    img_t = img.transpose((2,0,1))
    # convert to tensor
    return torch.from_numpy(img_t)

def predict(args, model):
    ''' Predict the class (or classes) of an image using a trained deep
    learning model.  '''
        
     # Process image
    img = process_image(args.i)
   
    if args.gpu:
        img = img.type(torch.gpu.FloatTensor)
    else:
        img = img.type(torch.FloatTensor)
    model_input = img.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    hi_probabilities, hi_probabilities_idx = probs.topk(args.top_k)
    hi_probabilities = hi_probabilities.detach().numpy().tolist()[0] 
    hi_probabilities_idx = hi_probabilities_idx.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    labels = [idx_to_class[idx] for idx in hi_probabilities_idx]
    
    flowers = [model.cat_to_name[idx_to_class[lab]] for lab in hi_probabilities_idx]
    
    return hi_probabilities, labels, flowers   



