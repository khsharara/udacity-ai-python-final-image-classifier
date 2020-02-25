#!/usr/bin/env python3                                              
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/13/2020                           
# REVISED DATE: 
# PURPOSE: Create a function that retrieves the following 5 command line inputs 
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the 5 inputs, then the default values are
#          used for the missing inputs. Command Line Arguments:
#     1. Image file that the model should infer, this is a required input
#     2. Checkpoint file that contains the saved model's details, this is a required input
#     3. The top k predicted categories as --top_k the program should print predictions for
#     4. The .json file that contains the category ID to name mapping as --category_names
#     5. Processor type the program should use as --gpu with default value 'cpu'
#
##
# Imports python modules
import argparse


def get_predict_input_args():
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
    parser.add_argument('image', type=str, help='The image file for the model to predict')
    parser.add_argument('checkpoint', type=str, help='The .pth checkpoint file that contains the trained model')
    parser.add_argument('--top_k', type=int, help='The top k predicted categories the program should print predictions for', default=3)
    parser.add_argument('--category_names', type=str, help='The .json file that contains the category ID to name mapping', default='cat_to_name.json')
    parser.add_argument('--gpu', help='Use only if the model will be trained on a gpu', action='store_true')
    
    return parser.parse_args()

#get_predict_input_args()