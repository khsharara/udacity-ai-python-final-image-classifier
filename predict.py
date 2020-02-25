#!/usr/bin/env python3
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/12/2020                                 
# REVISED DATE: 
# PURPOSE: Predict flower name from an image along with the probability of that name.
#
# Use argparse Expected Call with <> indicating expected user input:
#      Basic usage:
#          python predict.py </path/to/image> </path/to/checkpoint>
#          
#      Options:
#      Return top K most likely classes
#          python predict.py </path/to/image> </path/to/checkpoint> --top_k <3>
#      Use a mapping of categories to real names
#          python predict.py </path/to/image> </path/to/checkpoint> --category_names cat_to_name.json
#      Use GPU for inference
#          python predict.py </path/to/image> </path/to/checkpoint> --gpu
#             
##
import predict_utilities as utl
import json

from get_predict_input_args import get_predict_input_args

def main():

#     ### Load variables
#     image_path = 'flowers/test/19/image_06196.jpg' # The image to classify. TODO: Make dynamic based on argparser input.
#     checkpoint = 'checkpoints/checkpoint_1.pth' # The checkpointed model to use to classify. TODO: Make dynamic based on argparser input.
#     cat_to_name_json = 'cat_to_name.json' # JSON file that maps class values to category names. TODO: Make dynamic based on argparser input.
#     use_gpu = True # True if GPU should be used. TODO: Make dynamic based on argparser input.
#     num_top_k = 2 # Number of top K classes to print. TODO: Make dynamic based on argparser input.
    
    ### Get the user's input from the command line
    in_arg = get_predict_input_args()
    ### Load variables
    image_path = in_arg.image # The image to classify
    checkpoint = in_arg.checkpoint # The checkpointed model to use to classify
    cat_to_name_json = in_arg.category_names # JSON file that maps class values to category names
    use_gpu = in_arg.gpu # True if GPU should be used
    num_top_k = in_arg.top_k # Number of top K classes to print
    
    ### Load the json file that contains the category mappings into a dictionary
    with open(cat_to_name_json, 'r') as f:
        cat_to_name = json.load(f)
    
    ### Load the checkpoint
    model = utl.load_checkpoint(checkpoint)
    
    ### Make the prediction
    top_p, top_class = utl.predict(image_path, model, use_gpu, num_top_k)
    
    ### Display the output of the prediction
    utl.display_prediction(top_p, top_class, image_path, cat_to_name)


main()
