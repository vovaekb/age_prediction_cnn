Age prediction CNN Project
--------------------

This repository provides an implementation of person age prediction from image of face using CNN models.

Age prediction project allows to use two base CNN models (ModelNetV2 and ResNet50) from [Keras](https://keras.io/applications/). The models are trained via transfer learning, where ImageNet pre-trained CNNs are used and fine-tuned for the classification task.

Age prediction CNN is compatible with Python 3.6 and is distributed under the MIT license.

## Requirements

    # Python3.5+
    # Keras 2.2.5+

## Installation:

    # create age_prediction_cnn folder
    mkdir age_prediction_cnn
    cd age_prediction_cnn

    # clone and install locally
    git clone git@github.com:vovaekb/age_prediction_cnn.git

## Preparing data
Use some preprocessing script to crop out faces in images and obtain a person age. You should have a batch of image files with faces named in following format:

    <id>_<age>_<gender>.jpg

You can use [UTKFace](https://susanqq.github.io/UTKFace/). This dataset was used to train models in the Age prediction CNN. UTKFace dataset provides labels for both age and gender.

For UTKFace dataset you can use the script transform_dataset_names.py in root folder of the project. This script allows to prepare face crops in required format.
To run this script:

    --sample_dir
    <path_to_utkface_data>
    --output_dir
    <path_to_save_utkface_training_dataset>


## Running applications

When you are going to train the age predictor you just need to run python.

    python train_model.py --type
    <model_type>
    --train_sample_dir
    <train_sample_dir>
    --test_sample_dir
    <test_sample_dir>
    --model_path
    <model_path>
    --base_model
    <base_model_name>
    --img_dim
    <img_dim>
    --age_deviation
    <age_deviation>
    --load
    True
    --predict_gender
    True

Here **--type** denotes the type of NN model and can have two values ("classification" and "regression").

Parameters **--train_sample_dir** and **--test_sample_dir** specify path to train and test datasets accordingly.

**--model_path** specifies the path where model will be saved after training so predictor can load model from h5 file later.

**--base_model** means CNN architecture used (two values: MobileNetV2 and ResNet50).

**--img_dim** means dimension of input images for training (width, height), 128 on default.

**--age_deviation** specifies the deviation in age vector (in years), 5 on default

Optional parameters:

**--load means** that trained model will be loaded from h5 file rather than trained from scratch.

**--predict_gender** allows to apply gender classification in addition to age prediction.


If you run the application in training mode you should see something like this:

    Using TensorFlow backend.
    Initializing CNN model ...
    Starting training ...
    Epoch 1/13

    1/2 [==============>...............] - ETA: 8s - loss: 0.0947
    2/2 [==============================] - 13s 6s/step - loss: 0.0668 - val_loss: 0.1144
    Epoch 2/13
    ...

If you run the application with gender classification turned in you should see similar output with accuracy and loss for both age and gender.

Training accuracy and loss as well as validation accuracy will be printed in terminal.

If you run the application in loading mode you should see something like this:

    Using TensorFlow backend.
    Initializing CNN model ...
    
    Predicting for image 20170110224238891_10_0.jpg
    Predicted age: 0, true age: 10
    Predicted gender: M, true gender: M
    ...

If you run the application with gender classification turned on you should see similar output with predictions for both age and gender.

## Model training modes for age
### Classification mode
In classification mode chosen predictor builds a AGE vector represented as a histogram of a normal probability around the age value with deviation 5 years.

![](_readme/images/age_vector.png)

As loss and accuracy metrics [Earth mover's distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) and [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) are used respectively.


### Regression mode
In regression mode chosen predictor outputs single floating point value in range 0 to 1.0 representing age. [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) is used as loss.

## Gender classification
We also apply gender classification of a person on image. You can include this option using parameter --gender.

## Datasets
This project uses this dataset to train the prediction model:

[**UTKFace**](https://susanqq.github.io/UTKFace/)

## References

[Keras, Regression, and CNNs](https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/)
