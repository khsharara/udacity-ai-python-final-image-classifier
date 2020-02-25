#!/usr/bin/env python3
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/12/2020                                 
# REVISED DATE: 
# PURPOSE: Provides utility functions such as image preprocessing, loading checkpoints, etc to help the main predict script
#      
##
import torch
from torchvision import models
from PIL import Image
from pathlib import Path
import numpy as np

def load_checkpoint(filepath):
    """
    Loads a model from a trained checkpoint
    Parameters:
        filepath - [String] File path to the checkpoint
    Returns:
        model - [Pytorch Model] The checkpoint loaded as a model that can be used to make a prediction
    """
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True) 
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.classifier.load_sate_dict = checkpoint['classifier_state_dict']
    model.optimizer = checkpoint['optimizer_state_dict']
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    
    return model

def process_image(image):
    """
    Processes an image into a format that the model was trained on.
    Parameters:
        image - [String] File path to the image
    Returns:
        np_image - [NP Array] The fully processed image as a numpy array
    """
    # Open the image using PIL
    pil_image = Image.open(image)
    
    # Resize the image to 256x256 while maintining aspect ratio
    if pil_image.width > pil_image.height:
        resize_dim = (int(pil_image.width*256 / pil_image.height), 256)
    else:
        resize_dim = (256, int(pil_image.height*256 / pil_image.width))
    
    pil_image = pil_image.resize(resize_dim)
    
    # Crop image to center 224 pixles
    crop_box_dim = 224
    left = (pil_image.width - crop_box_dim)/2
    top = (pil_image.height - crop_box_dim)/2
    right = pil_image.width - (pil_image.width - crop_box_dim)/2
    bottom = pil_image.height - (pil_image.height - crop_box_dim)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Update color channels
    np_image = np.array(pil_image)
    np_image_means = np.array([0.485, 0.456, 0.406])
    np_image_stddev = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255 - np_image_means) / np_image_stddev
    
    # PIL images and numpy arrays have color channels in the 3rd dimension
    # Transpose them to first dimension to match what PyTorch expects
    np_image = np_image.transpose((2,0,1))

    return np_image

def predict(image_path, model, use_gpu=False, top_k=5):
    """
    Predict the top class (or classes) of an image using the trained deep learning model.
    Parameters:
        image_path - [Text] File path to the image the model should make the prediciton on
        model - [Pytorch model] The model that should be used to make the prediction
        use_gpu - [Boolean] True if GPU should be used
        top_k - [Integer] Number of top classes the model should return as part of the prediction
    Returns:
        top_p - [List Floats] The prediction for each top_k class
        top_class - [Integer] The class ID of the prediction
    """
    # Determine if pytorch should run on a GPU or CPU depending on user input
    device = torch.device("cuda" if use_gpu else "cpu")
    
    # Switch evaluation mode 'on'
    model.eval()
    
    # Preprocess the image
    img = torch.FloatTensor([process_image(image_path)])
    
    # Run the inference
    with torch.no_grad():
        model.to(device)
        logps = model.forward(img.to(device))
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(top_k, dim=1)
        
        class_to_idx = model.class_to_idx
        idx_to_class = {str(value):int(key) for key, value in class_to_idx.items()}

        top_p = top_p.cpu().numpy().flatten()
        top_class = np.array([idx_to_class[str(idx)] for idx in top_class.cpu().numpy().flatten()])
        
    return top_p, top_class

def display_prediction(top_p, top_class, image_path, cat_to_name):
    """
    Prints the prediction results
    Parameters:
        top_p - [List Floats] The prediction for each top_k class
        top_class - [Integer] The class ID of the prediction
        image_path - [Text] File path to the image the model should make the prediciton on
        cat_to_name - [Dictionary] Mapings between category ID and category name
    Returns:
        none - prints output to the terminal
    """
    # Create a Path object that defines where the image lives. This will help categorize the image by using the parent directory ID.
    path = Path(image_path)

    # Get the names of each class
    names = [cat_to_name[str(c)] for c in top_class]

    # Get the actual image category name
    image_title = cat_to_name[str(path.parent.name)]
    
    # Print the actual image name and the model's top k predictions
    print("The actual image is a '{}' and here are the model's top {} prediction(s):".format(image_title, len(top_p)))
    counter = 1
    for name, prediction in zip(names, top_p):
        print('{}) {:.1%} {}'.format(counter, prediction, name))
        counter += 1
    
