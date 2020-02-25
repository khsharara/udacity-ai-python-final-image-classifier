#!/usr/bin/env python3
# PROGRAMMER: Khalid S.
# DATE CREATED: 02/12/2020                                 
# REVISED DATE: 
# PURPOSE: Functions used to define the model
##
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def train_model(train_loader, validation_loader, arch, learning_rate, num_hidden_units, epochs, use_gpu):
    """
        Trains the model on the training dataset and validate on the validation dataset.
        Parameters:
            train_loader - loads the training data
            validation_loader - loads the validation data
            arch - Model architecture to use
            learning_rate - The learning rate the model should use
            num_hidden_units - The number of hidden units should use
            epochs - Number of epochs to train with
            use_gpu - True if GPU should be used
        Returns:
            device - The device type (gpu, or cpu) that pytorch used
            model - The actual NN model
            criterion - The criterion used to score the model
            optimizer - The optimizer used to optimize the model
        """
    print("Initializing the model...")
    # #### Define the network
    # Determine if pytorch should run on a GPU or CPU depending on user input
    device = torch.device("cuda" if use_gpu else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # This will automatically use GPU if it's available

    # Select model architecture based on user input
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        number_input_features = model.classifier[0].in_features # Pulls the number of input features, 25088, from the pretrained model
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        number_input_features = model.classifier.in_features  # Pulls the number of input features, 1024, from the pretrained model
    else:
        print("You must select a valid architecture, either: vgg or densenet")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(number_input_features, num_hidden_units), 
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(num_hidden_units, 102),
                                         nn.LogSoftmax(dim=1))    

    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) # TODO: update logic to select lr based on flag

    model.to(device);


    # ### Train the network
    train_losses, validation_losses = [], []
    for epoch in range(epochs):
        train_loss = 0
        model.train()
        for inputs, labels in train_loader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # DONE: need the validation loss and accuracy to be displayed
        else:
            validation_loss = 0
            accuracy = 0

            # Turn off gradients for validation to save memory and computation
            with torch.no_grad():
                model.eval()
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    validation_loss += criterion(logps, labels)

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

                    train_losses.append(train_loss/len(train_loader))
                    validation_losses.append(validation_loss/len(validation_loader))

                    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                          "Training Loss: {:.3f}".format(train_loss/len(train_loader)),
                          "Validation Loss: {:.3f}".format(validation_loss/len(validation_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))

    return device, model, criterion, optimizer, number_input_features

def validate_on_test_dataset(device, model, criterion, test_loader):
        """
        Validate the model on the test dataset.
        Parameters:
            model - The model to be used
            test_loader - The test dataset loader
        Returns:
            none - prints output
        """
        # ### Do validation on the test set
        validation_loss = 0
        accuracy = 0
        # turn off gradient calculations to speed up computation
        with torch.no_grad():
            model.eval()
            for inputs, labels in test_loader:
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                log_ps = model(inputs)
                test_loss = criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        print("Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)))

def save_checkpoint(save_dir, model, arch, train_data, optimizer, epochs, number_input_features):
    """
    Saves the state of the model in a checkpoint file
    Parameters:
        save_dir - Directory to save the checkpoint to
        model - The trained model
        train_data - The training data file
        optimizer - The optimizer that was used to train
        epochs - The number of epochs used to train
    Returns:
        none - Prints a statement when the checkpoint completes
    """
    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    model.optimizer = optimizer
    model.input_size = number_input_features
    model.output_size = 102
    model.arch = arch
    model.epochs = epochs
    checkpoint = {"input_size": model.input_size,
                  "output_size": model.output_size,
                  "arch": model.arch,
                  "epochs": model.epochs,
                  "class_to_idx": model.class_to_idx,
                  "classifier": model.classifier,
                  "classifier_state_dict": model.classifier.state_dict(),
                  "optimizer_state_dict": model.optimizer.state_dict()}
    
    torch.save(checkpoint, save_dir + "/checkpoint.pth")
    print('Checkpoint saved in {}'.format(save_dir + "/checkpoint.pth"))