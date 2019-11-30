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

def load_data(train_dir, valid_dir, test_dir ):
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
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

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
    """ Train NN

    :model: 
    :epochs: TODO
    :data: TODO
    :criterion: TODO
    :returns: TODO

    """
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


def main(initial_timestamp, args): 
    """
    train.py  should have cli arguments 
        python train.py <data_dir> 
        python train.py --save_dir <save_dir>
        python train.py --arch "vgg13"
        python train.py --learning_rate 0.01
        python train.py --hidden_units 512
        python train.py --epochs 20
        python train.py --gpu

    """

    # Setting up Logging facility
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            stream = sys.stdout,
            level = logging.DEBUG,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(thread)d' +
            ' - %(message)s',
            )

    # Directory with images
    data_dir = args.data_dir

    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'


    # Create Model
    #   - Use pretrained network Building and training the classifier
    model = models.__dict__[args.arch](pretrained=True)
    logger.info(f'Using pre-trained model {args.arch}')
 
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
        device = "cpu"
    logger.info(f'Using {device.type} for calculations...')

    # Load Data & make Trainsformations
    data = load_data(train_dir, valid_dir, test_dir)
    logger.info(f'Load Data from {train_dir}, {valid_dir}, {test_dir} ..done!')
   
    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    num_of_fw_classes = len(cat_to_name.keys())
    logger.info(f'Read json file for Label mapping. Number of classes: ' +\
            f'{num_of_fw_classes}')

    # Freeze parameters so we don't backprop through them
    logger.info(f'Freeze parameter of the model.. ')
    for param in model.parameters():
        param.requires_grad = False
    # Change classifier 
    logger.info(f'Change model classifier.. ')
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 1000)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(1000, num_of_fw_classes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
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
                f'{args.saved_pth_file}')

    logger.info(f'Start training NN ...')
    # Save the checkpoint and attach class_to_index to the model
    model = train(model, args.epochs, data, optimizer, criterion, device,
            logger)
    torch.save(model.state_dict(), saved_pth_file)
    logger.info(f'Saving NN to file...')

    # TODO: predict.py
    print(" --- [{}] --- End Timestamp --- ".format(datetime.now() - initial_timestamp))


if __name__ == '__main__':
    # Start timesamp
    initial_timestamp = datetime.now()
    print(" --- [{}] --- Start Timestamp --- ".format(initial_timestamp))
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    # Configuration optinos
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--save_dir', default='./checkpoints',
            help='path to saved checkpoint dir')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
            choices=model_names, 
            help='model architecture: ' + ' | '.join(model_names) +\
                    ' (default: vgg11)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate default 0.01')
    parser.add_argument('--hidden_units', dest='hidden units',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--gpu',  help='use GPU', action='store_true')
    
    cli_args = parser.parse_args()

    sys.exit(main(initial_timestamp, cli_args))

