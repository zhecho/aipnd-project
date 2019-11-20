#!/usr/bin/env python
# coding: utf-8


import json, time
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os.path import isfile

data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
saved_pth_file = 'flowers_saved_checkpoint.pth'

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
#     transforms.CenterCrop(224),                                                
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  
valid_transforms = transforms.Compose([                                         
    transforms.Resize((224,224)),
#     transforms.CenterCrop(224),                                               
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

num_of_fw_classes = len(cat_to_name.keys())

# We can load in a model such as
# [DenseNet](http://pytorch.org/docs/0.3.0/torchvision/models.html#id5). Let's
# print out the model architecture so we can see what's going on.

model = models.densenet121(pretrained=True)

# This model is built out of two main parts, the features and the classifier.
# The features part is a stack of convolutional layers and overall works as a
# feature detector that can be fed into a classifier. The classifier part is a
# single fully-connected layer `(classifier): Linear(in_features=1024,
# out_features=1000)`. This layer was trained on the ImageNet dataset, so it
# won't work for our specific problem. That means we need to replace the
# classifier, but the features will work perfectly on their own. In general, I
# think about pre-trained networks as amazingly good feature detectors that can
# be used as the input for simple feed-forward classifiers.

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

# Check and load saved dict state if file exist
if isfile(saved_pth_file):
    state_dict = torch.load(saved_pth_file)
    model.load_state_dict(state_dict)


epochs = 2
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    start_ts = datetime.now()
    model.train()
    for images, labels in trainloader:
        steps += 1
        images, labels = images.to(device), labels.to(device) 
        
        # Flatten images into a 224x224 = 50167   long vector
        #images.resize_(images.size()[0], 50176)
        
        optimizer.zero_grad()
        
        # from IPython.core import debugger; debug = debugger.Pdb().set_trace; debug()
        # from IPython import embed; embed()
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
                test_loss, accuracy = validation(model, validloader, criterion)
                        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
                  "Runtime: {}".format(datetime.now() - start_ts), )
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
            torch.save(model.state_dict(), saved_pth_file )


