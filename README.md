# udacity-ai-python-final-image-classifier
This was was my final project as part of the AI Programming with Python nano degree offered through Udacity

This project has two main programs that accept parameters through the command line - train.py and predict.py.

1) train.py
Trains a neural network on a dataset and saves the model as a checkpoint.
Usage below <> indicates expected user input.
      Basic usage:
      Prints out training loss, validation loss, and validation accuracy as the network trains
          python train.py <data_dir>
          
      Options:
      Set directory to save checkpoints
          python train.py data_dir --save_dir <save_directory>
      Choose architecture
          python train.py data_dir --arch <vgg>
      Set hyperparameters
          python train.py data_dir --learning_rate <0.01> --hidden_units <512> --epochs <20>
      Use GPU for training
          python train.py data_dir --gpu
             
   Example call:
    python check_images.py data_dir --arch <vgg> --dogfile <dognames.txt>


2) predict.py
Infer object name from an image along with the probability that the object was predicted correctly.
The prediction uses a checkpointed model that is saved on disk.
Usage below <> indicates expected user input.
      Basic usage:
          python predict.py </path/to/image> </path/to/checkpoint>
          
      Options:
      Return top K most likely classes
          python predict.py </path/to/image> </path/to/checkpoint> --top_k <3>
      Use a mapping of categories to real names
          python predict.py </path/to/image> </path/to/checkpoint> --category_names cat_to_name.json
      Use GPU for inference
          python predict.py </path/to/image> </path/to/checkpoint> --gpu
             
