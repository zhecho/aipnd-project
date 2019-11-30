#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import json, time, sys
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os


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
    logger.info(f'Read json file for Label mapping. Number of classes: ' +\
            f'{num_of_fw_classes}')
    return num_of_fw_classes, cat_to_name

def create_model(args):
    """ Create Model from pretrained NN

    :args: CLI arguments
    :returns: model of NN

    """
    #   - Use pretrained network Building and training the classifier
    model = models.__dict__[args.arch](pretrained=True)
    return model


def change_classifier(args, model, num_of_fw_classes):
    """ Change classifier model to number of classes """
    # Freeze parameters so we don't backprop through them
    logger.info(f'Freeze parameter of the model.. ')
    for param in model.parameters():
        param.requires_grad = False
    # Change classifier 
    new_in_features = model.classifier[0].in_features
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
 

def main(args, logger): 
    """
    Options train.py should have cli arguments 
        python train.py <data_dir> 
        python train.py --save_dir <save_dir>
        python train.py --arch "vgg13"
        python train.py --learning_rate 0.01
        python train.py --hidden_units 512
        python train.py --epochs 20
        python train.py --gpu """


    # Directory with images
    data_dir = args.data_dir

    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    # Create Model
    model = create_model(args)
    logger.info(f'Using pre-trained model {args.arch}')

    # Label Mapping
    num_of_fw_classes, cat_to_name = label_map(args)

    # Change model classifier for flower classes
    model = change_classifier(args, model, num_of_fw_classes)

    # Check and use Saved model checkpoints dir & files
    # optionally resume from a checkpoint
    check_dir(args.save_dir)
    saved_pth_file = args.save_dir +\
            '/flowers_saved_'+ args.arch + '_checkpoint.pth'
    if os.path.isfile(saved_pth_file):
        state_dict = torch.load(saved_pth_file)
        model.load_state_dict(state_dict)
        logger.info(f'Loading Checkpoint file {saved_pth_file}')
    else:
        logger.info(f'Checkpoint file {saved_pth_file} not found..')

    # Use Nvidia chips if exists on the system
    if args.gpu: 
        # Check if GPU exist
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = str("cpu")
    logger.info(f'Using {device} for calculations...')

    # Load Data & make Trainsformations
    data = load_data(train_dir, valid_dir, test_dir, args.batch_size)
    logger.info(f'Load Data from {train_dir}, {valid_dir}, {test_dir} ..done!')
   
    criterion = nn.NLLLoss()
    logger.info(f'Using NLLLoss')
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    logger.info(f'Using optimizer {optimizer.__repr__()} ')
    model.to(device)

    # Loading checkpoint
    # Check and load saved dict state if file exist
    if os.path.isfile(saved_pth_file):
        state_dict = torch.load(saved_pth_file)
        model.load_state_dict(state_dict)
        logger.info(f'Loading saved model checkpoints from ' + \
                f'{saved_pth_file}')

    logger.info(f'Start training NN ...')
    # Save the checkpoint and attach class_to_index to the model
    model = train(model, args.epochs, data, optimizer, criterion, device, logger)
    torch.save(model.state_dict(), saved_pth_file)
    logger.info(f'Saving NN to file...')

if __name__ == '__main__':

    # Setting up Logging facility
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            stream = sys.stdout,
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(thread)d' +
            ' - %(message)s',)

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    # Configuration optinos
    parser = argparse.ArgumentParser(description='PyTorch Image Trainer')
    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--save_dir', default='./checkpoints',
            help='path to saved checkpoint dir (default: "./checkpoints" )')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
            choices=model_names, help='model architecture: ' + \
                    ' | '.join(model_names) + ' (default: vgg11)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
            help='number of total epochs to run')
    parser.add_argument('--category_names', default='./cat_to_name.json',
            help='File with class mapping (default: "./cat_to_name.json"')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int,
            default=1024, help='Hidden Units (default: "1024" )')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
            metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
            metavar='LR', help='initial learning rate default 0.001')
    parser.add_argument('--gpu',  help='use GPU', action='store_true') 

    args = parser.parse_args()

    # Run main
    sys.exit(main(args=args, logger=logger))

