#!/usr/bin/env python3                                              
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/13/2020                           
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 7 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 3 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image directory as --input_dir with default value 'flowers'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Learning rate as --learning_rate with default value 0.01
#     4. Number of hidden units as --hidden_units with default value of 512
#     5. Number of epochs as --epochs with default value 20
#     6. Processor type the program should use as --gpu with default value 'cpu'
#     7. Location to save checkpoint as --save_dir with default value 'checkpoints'
#
##
# Imports python modules
import argparse

def get_train_input_args():
    """
    Create a function that retrieves the following 7 command line inputs 
    from the user using the Argparse Python module. If the user fails to 
    provide some or all of the 3 inputs, then the default values are
    used for the missing inputs. Command Line Arguments:
     1. Image directory as --input_dir with default value 'flowers'
     2. CNN Model Architecture as --arch with default value 'vgg'
     3. Learning rate as --learning_rate with default value 0.01
     4. Number of hidden units as --hidden_units with default value of 512
     5. Number of epochs as --epochs with default value 20
     6. Processing unit as --gpu with default value 'cpu'
     7. Location to save checkpoint as --save_dir with default value 'checkpoints'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser("Enter the arguments to feed the model of choice.")
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('dir', type=str, help='Image folder that contains the training, test, and validation data', default='flowers')
    parser.add_argument('--arch', type=str, help='CNN Model Architecture, vgg or densenet', default='vgg')
    parser.add_argument('--learning_rate', type=float, help='The learning rat the model should use', default=0.01)
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units in the model', default=256) # TODO: unsure if this has been programmed for
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the model on', default=5)
    parser.add_argument('--save_dir', type=str, help='The directory the checkpoint should be saved to', default='checkpoints')
    parser.add_argument('--gpu', help='Use only if the model will be trained on a gpu', action='store_true')
    
    return parser.parse_args()

#get_train_input_args()