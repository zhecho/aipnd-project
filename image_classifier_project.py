#!/usr/bin/env python
# coding: utf-8
import json, time, sys
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os.path import isfile



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


# Implement a function for the validation pass
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:

        # images.resize_(images.shape[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train(model, epochs, data, optimizer, criterion, device):
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
                            model, data['validloader'], criterion)
                            
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
                      "Runtime: {}".format(datetime.now() - start_ts), )
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()
    return model

def main(initial_timestamp):

    # Configuration optinos
    data_dir = 'flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    saved_pth_file = 'flowers_saved_checkpoint.pth'
    learning_rate = 0.001
    epochs = 1

    # Load Data & make Trainsformations
    data = load_data(train_dir, valid_dir, test_dir)

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    num_of_fw_classes = len(cat_to_name.keys())

    # Build and traing NN
    #   - Use pretrained network Building and training the classifier
    model = models.densenet121(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 500)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(500, num_of_fw_classes)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    # Use Nvidia chips if exists on the system
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    # Loading checkpoint
    # Check and load saved dict state if file exist
    if isfile(saved_pth_file):
        state_dict = torch.load(saved_pth_file)
        model.load_state_dict(state_dict)

    # Testing NN - do validation on the test set
    model = train(model, epochs, data, optimizer, criterion, device)

    # Save the checkpoint and attach class_to_index to the model
    torch.save(model.state_dict(), saved_pth_file)

    # Loading checkpoint

    # Inference for classification 

    # Image Preprocessing

    # Class Prediction

    # Sanity Checking
    print(" --- [{}] --- End Timestamp --- ".format(datetime.now() - initial_timestamp))

if __name__ == '__main__':
    # Start timesamp
    initial_timestamp = datetime.now()
    print(" --- [{}] --- Start Timestamp --- ".format(initial_timestamp))
    sys.exit(main(initial_timestamp))
