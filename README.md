Age prediction CNN Project
--------------------

This repository provides an implementation of person age prediction from image of face using CNN models.

Age prediction project allows to use two base CNN models (ModelNetV2 and ResNet50) from [Keras](https://keras.io/applications/). The models are trained via transfer learning, where ImageNet pre-trained CNNs are used and fine-tuned for the classification task.

Age prediction CNN is compatible with Python 3.6 and is distributed under the MIT license.

## Installation (dev):

    # we need python3.5+

    # create idtotal folder
    mkdir age_prediction_cnn
    cd age_prediction_cnn

    # clone and install locally
    git clone git@github.com:vovaekb/age_prediction_cnn.git

## Preparing data
Use some preprocessing script to crop out faces in images and obtain a person age. You should have a batch of image files with faces named in following format:

    <id>_<age>.jpg

You can use [WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). This dataset was used to train models in the Age prediction CNN.

For face detection you can use [Ximilar face detector API](https://docs.ximilar.com/services/face_detection/).

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
    --load
    True

Here --type denotes the type of NN model and can have two values ("classification" and "regression"). When classification type chosen predictor builds a AGE vector represented as a normal probability histogram and Earth mover distance loss used. When regression type chosen predictor outputs single floating point value for age.
Parameters --train_sample_dir and --test_sample_dir specify path to train and test datasets accordingly.
--model_path specifies the path where model will be saved after training so predictor can load model from h5 file later.
--base_model means CNN architecture used (two values: MobileNetV2 and ResNet50).
--img_dim means dimension of input images for training (width, height), e.g. 128.
Optional parameter --load means that trained model will be loaded from h5 file rather than trained from scratch.

If you run the application in training mode you should see something like this:

    Using TensorFlow backend.
    Initializing CNN model ...
    Starting training ...
    Epoch 1/13

    1/2 [==============>...............] - ETA: 8s - loss: 0.0947
    2/2 [==============================] - 13s 6s/step - loss: 0.0668 - val_loss: 0.1144
    Epoch 2/13

    1/2 [==============>...............] - ETA: 1s - loss: 0.1524
    2/2 [==============================] - 6s 3s/step - loss: 0.1755 - val_loss: 0.1844
    Epoch 3/13
    ...

Training accuracy and loss as well as validation accuracy will be printed in terminal.

If you run the application in loading mode you should see something like this:

    Using TensorFlow backend.
    Initializing CNN model ...
    True label: 48
    Predicted label: 8.177467435598373
    True label: 70
    Predicted label: 99.9997615814209
    ...


## Datasets
This project uses this dataset to train the prediction model:

[**WIKI**](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
