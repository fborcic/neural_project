Dependencies:
    python 3.8
    pytorch
    pillow (PIL)

Files:
    net.py - CNN model definition
    train.py - CNN training code
    preprocessor.py - numpy feature extraction code
    read_number.py - main entry point

    test_images - a set of images for testing
    parameters.tf - parameters of the CNN trained for 10 epochs

The neural network can be trained by running:
    python3 train.py

The hyperparameters can be adjusted by modifying the constants at the start of the script. After
training, the parameters are saved to parameters.tf.

To read a number, run:
    python3 read_number.py [path_to_image]

If you want to save the intermediate results of image processing, run:
    python3 read_number.py [path_to_image] debug
After doing so, the intermediate results will be saved to the current working directory.
