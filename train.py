#!/usr/bin/env python3
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/12/2020                                 
# REVISED DATE: 
# PURPOSE: Train a new network on a dataset and save the model as a checkpoint.
#
# Use argparse Expected Call with <> indicating expected user input:
#      Basic usage:
#      Prints out training loss, validation loss, and validation accuracy as the network trains
#          python train.py data_dir
#          
#      Options:
#      Set directory to save checkpoints
#          python train.py data_dir --save_dir <save_directory>
#      Choose architecture
#          python train.py data_dir --arch <vgg>
#      Set hyperparameters
#          python train.py data_dir --learning_rate <0.01> --hidden_units <512> --epochs <20>
#      Use GPU for training
#          python train.py data_dir --gpu
#             
#   Example call:
#    python check_images.py data_dir --arch <vgg> --dogfile <dognames.txt>
##

import train_utilities as utl
import train_model_definition as tmd
from get_train_input_args import get_train_input_args

def main():
    ### Get the user's input from the command line
    in_arg = get_train_input_args()
    
    ### Load variables
    # Which directories should we load and save to?
    data_dir = in_arg.dir # Path to the parent directory. TODO: Make dynamic based on argparser input. 
    save_dir = in_arg.save_dir # Path to the directory to save the checkpoint in. TODO: Make dynamic based on argparser input. 
    
    # Which parameters should we use to train the model?
    arch = in_arg.arch # Model architecture to use. TODO: Make dynamic based on argparser input.
    learning_rate = in_arg.learning_rate # The learning rate the model should use. TODO: Make dynamic based on argparser input.
    num_hidden_units = in_arg.hidden_units # The number of hidden units should use. TODO: Make dynamic based on argparser input.
    epochs = in_arg.epochs # Number of epochs to train with. # TODO: Make dynamic based on argparser input.
    use_gpu = in_arg.gpu # True if GPU should be used. # TODO: Make dynamic based on argparser input.
    
    
    ### Run utility functions to load the data
    # Define the directories that contain each dataset
    train_dir, validation_dir, test_dir = utl.define_dataset_directories(data_dir) # TODO: Make dynamic based on argparser input
    # Define the transformations to apply to each dataset
    train_transforms, validation_transforms, test_transforms = utl.define_data_transforms()
    # Load the datasets with the transforms applied to them
    train_data, validation_data, test_data = utl.load_datasets(train_dir, validation_dir, test_dir, train_transforms, validation_transforms, test_transforms)
    # Define the dataloaders for each dataset
    train_loader, validation_loader, test_loader = utl.define_dataloaders(train_data, validation_data, test_data)
    
    ### Train the model
    device, model, criterion, optimizer, number_input_features = tmd.train_model(train_loader, validation_loader, arch, learning_rate, num_hidden_units, epochs, use_gpu)
    
    ### Validate the model on the test dataset
    tmd.validate_on_test_dataset(device, model, criterion, test_loader)
    
    ### Save the state of the model in a checkpoint
    tmd.save_checkpoint(save_dir, model, arch, train_data, optimizer, epochs, number_input_features)  

main()