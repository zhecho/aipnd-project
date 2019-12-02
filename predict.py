#!/usr/bin/env python
# coding: utf-8
import argparse
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

from train import ( create_model, check_dir, load_model, label_map)


def process_image(image):
    ''' Scales, crops, and t a PIL image for a PyTorch model, returns an Numpty
    array '''
    # Load, resize, crop image
    img = Image.open(image)
    img = img.resize((256,256))
    img = img.crop((16,16,240,240))
    
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
    hi_probabilities, hi_probabilities_idx = probs.topk(args.topk)
    hi_probabilities = hi_probabilities.detach().numpy().tolist()[0] 
    hi_probabilities_idx = hi_probabilities_idx.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    labels = [idx_to_class[idx] for idx in hi_probabilities_idx]
    
    flowers = [cat_to_name[idx_to_class[lab]] for lab in hi_probabilities_idx]
    
    return hi_probabilities, labels, flowers   
    

def main(args, logger):
    """ CLI args to predict.py  
        python predict.py -i <path_to_image> 
        python predict.py --category_names <cat_to_name.json>
        python predict.py --save_dir <save_dir>
        python predict.py --top_k <top k classes>
        python predict.py --gpu
    """
    # Create Model
    model = create_model(args)

    # Label Mapping
    num_of_fw_classes, cat_to_name = label_map(args)
    logger.info(f'Read json file for Label mapping. Number of classes: ' +\
            f'{num_of_fw_classes}')
   
    # Check and load saved checkpoint file 
    check_dir(args.save_dir)
    saved_pth_file = args.save_dir +\
            '/flowers_saved_'+ args.arch + '_checkpoint.pth'
    if os.path.isfile(saved_pth_file):
        model = load_model(args, saved_pth_file, num_of_fw_classes, logger)
        logger.info(f'Loading Checkpoint file {saved_pth_file}')
    else:
        logger.error(f'Checkpoint file {saved_pth_file} not found..')
        sys.exit()

    # Prediction
    probs, labs, flowers = predict(args, model) 
    logger.info(f'Prediction: \n Probs: {probs} ' + \
            f'Labels: {labs} \n' + \
            f'Flowers: {flowers}')


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
    parser.add_argument('-i', metavar='PATH', help='path to image file'),
    parser.add_argument('--save_dir', default='./checkpoints',
            help='path to saved checkpoint dir')
    parser.add_argument('--category_names', default='./cat_to_name.json',
            help='File with class mapping')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
            choices=model_names, help='model architecture: ' + \
                    ' | '.join(model_names) + ' (default: vgg11)')
    parser.add_argument('--top_k', default=5, type=int, metavar='N',
            help='top k classes ')
    parser.add_argument('--hidden_units', dest='hidden_units', type=int,
            default=1024, help='Hidden Units')
    parser.add_argument('--gpu', help='use GPU') 

    args = parser.parse_args()

    # Run main
    sys.exit(main(args=args, logger=logger))

