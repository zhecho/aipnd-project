#!/usr/bin/env python
# coding: utf-8
import argparse, logging, sys, os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from functions import (
        create_model, load_model, label_map, check_dir, load_data,
        validation, train, change_classifier
        )

logger = logging.getLogger(__name__)


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
    logger.info(f'Read json file for Label mapping. Number of classes: ' +\
            f'{num_of_fw_classes}')

    # Use Nvidia chips if exists on the system
    if args.gpu: 
        # Check if GPU exist
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = str("cpu")
    logger.info(f'Using {device} for calculations...')
    
    # Check and use Saved model checkpoints dir & files
    # optionally resume from a checkpoint
    check_dir(args.save_dir)
    saved_pth_file = args.save_dir +\
            '/flowers_saved_'+ args.arch + '_' + str(device) + '_checkpoint.pth'
    if os.path.isfile(saved_pth_file):
        model = load_model(saved_pth_file)
        logger.info(f'Loading Checkpoint file {saved_pth_file}')
    else:
        logger.info(f'Checkpoint file {saved_pth_file} not found..')

    # Load Data & make Trainsformations
    data = load_data(train_dir, valid_dir, test_dir, args.batch_size)
    logger.info(f'Load Data from {train_dir}, {valid_dir}, {test_dir} ..done!')
   
    # Adding class_to_idx in the model
    model.class_to_idx = data['train_data'].class_to_idx
    # Change model classifier for flower classes
    checkpoint_dict = {
        'weights': model.state_dict(),
        'model_name': args.arch,
        'hidden_units': args.hidden_units,
        'output_units': num_of_fw_classes,
        'class_to_idx': model.class_to_idx }
    model = change_classifier(model, checkpoint_dict)

    # Loss
    criterion = nn.NLLLoss()
    logger.info(f'Using NLLLoss')
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    logger.info(f'Using optimizer {optimizer.__repr__()} ')
    model.to(device)

    logger.info(f'Start training NN ...')
    model = train(model, args.epochs, data, optimizer, criterion, device, logger)
    # Save the checkpoint and attach class_to_index to the model
    logger.info(f'Save model with some extra options')
    checkpoint_dict = {
            'weights': model.state_dict(),
            'model_name': args.arch,
            'hidden_units': args.hidden_units,
            'output_units': num_of_fw_classes,
            'class_to_idx': model.class_to_idx }
    torch.save(checkpoint_dict, saved_pth_file)

    # torch.save(model.state_dict(), saved_pth_file)
    logger.info(f'Saving NN to file...')

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

    supported_nets = list()
    for net in model_names:
        # Filter out only supported networks 
        if net.startswith(('vgg','densenet','resnet','alexnet')):
            supported_nets.append(net)

    # Configuration optinos
    parser = argparse.ArgumentParser(description='PyTorch Image Trainer')
    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--save_dir', default='./checkpoints',
            help='path to saved checkpoint dir (default: "./checkpoints" )')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg11',
            choices=supported_nets, help='model architecture: ' + \
                    ' | '.join(supported_nets) + ' (default: vgg11)')
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

