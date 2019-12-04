#!/usr/bin/env python
# coding: utf-8
import os, argparse, logging, sys
import torch
from torchvision import datasets, transforms, models

from functions import (
        create_model, load_model, label_map, check_dir, load_data,
        process_image, predict
        )


logger = logging.getLogger(__name__)

def main(args):
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
    # we need device str() in order to choose checkpoint file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_dir(args.save_dir)
    saved_pth_file = args.save_dir +\
            '/flowers_saved_'+ args.arch + '_' + str(device) + '_checkpoint.pth'

    if os.path.isfile(saved_pth_file):
        model = load_model(saved_pth_file)
        logger.info(f'Loading Checkpoint file {saved_pth_file}')
    else:
        logger.error(f'Checkpoint file {saved_pth_file} not found..')
        sys.exit()

    # Adding class_to_idx in the model
    # Directory with images
    data_dir = args.data_dir
    train_dir = data_dir + 'train'
    valid_dir = data_dir + 'valid'
    test_dir = data_dir + 'test'

    # Load Data & make Trainsformations
    data = load_data(train_dir, valid_dir, test_dir, args.batch_size)
    logger.info(f'Load Data from {train_dir}, {valid_dir}, {test_dir} ..done!')
    model.class_to_idx = data['train_data'].class_to_idx
    model.cat_to_name = cat_to_name

    # Prediction
    probs, labs, flowers = predict(args, model) 
    logger.info(f'Prediction: \n Probs: {probs} ' + \
            f'Labels: {labs} \n' + \
            f'Flowers: {flowers}')


if __name__ == '__main__':

    # Setting up Logging facility
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
    parser.add_argument('-i', metavar='PATH',
            help='path to image file', required=True),
    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
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
    parser.add_argument('--gpu', default=False, help='enable gpu support',
            action = 'store_true') 
    parser.add_argument('-b', '--batch-size', default=64, type=int,
            metavar='N', help='mini-batch size (default: 64)')

    args = parser.parse_args()

    # Run main
    sys.exit(main(args=args))

