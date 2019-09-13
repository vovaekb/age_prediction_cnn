import os
import cv2
import numpy as np
import argparse
from facematch.age_prediction.handlers.data_generator import DataGenerator
from keras.models import model_from_json
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from facematch.age_prediction.models.age_classification_net import AgeClassificationNet
from facematch.age_prediction.models.age_regression_net import AgeRegressionNet
from facematch.age_prediction.utils.metrics import earth_movers_distance, age_mae
from facematch.age_prediction.utils.utils import build_age_vector

IMG_SHAPE = (128, 128, 3)
EPOCHS = 13

parser = argparse.ArgumentParser()

get_custom_objects().update({"earth_movers_distance": earth_movers_distance, "age_mae": age_mae})


def train_model():
    parser.add_argument("-d", "--train_sample_dir", help="Path to raw images for training")
    parser.add_argument("-f", "--face_directory", help="Path to images of faces")
    parser.add_argument("-v", "--test_sample_dir", help="Path to images for validation")
    parser.add_argument("-w", "--model_path", help="Path to model JSON and weights")
    parser.add_argument("-s", "--img_dim", type=int, help="Dimension of input images for training (width, height)")
    parser.add_argument("-t", "--type", type=str, help="Type of model to use (regression, classification)")
    parser.add_argument("-b", "--base_model", type=str, help="Base model to use in the NN model (resnet, mobile_net)")
    parser.add_argument("-l", "--load", default=False, type=bool, help="Load model from file")
    args = vars(parser.parse_args())

    img_dim = args["img_dim"]

    print("Initializing CNN model ...")

    if args["type"] == "classification":
        age_model = AgeClassificationNet(args["base_model"], IMG_SHAPE)
    else:
        # REGRESSION MODEL
        age_model = AgeRegressionNet(args["base_model"], IMG_SHAPE)

    if not args["load"]:
        age_model.build()

        train_generator = DataGenerator(
            args,
            samples_directory=args["train_sample_dir"],
            batch_size=10,
            generator_type="train",
            basemodel_preprocess=age_model.preprocessing_function(),
            shuffle=True,
        )
        validation_generator = DataGenerator(
            args,
            samples_directory=args["test_sample_dir"],
            batch_size=10,
            generator_type="test",
            basemodel_preprocess=age_model.preprocessing_function(),
            shuffle=False,
        )

        # Compile model
        age_model.compile()

        # Fit generator and start training
        print("Starting training ...")

        # Train without loop and prediction monitoring
        age_model.model.fit_generator(
            generator=train_generator, validation_data=validation_generator, epochs=EPOCHS, verbose=1
        )

        # save the entire model
        model_file = os.path.join(args["model_path"], "model.h5")
        age_model.model.save(model_file)
    else:
        # Load model from file
        model_file = os.path.join(args["model_path"], "model.h5")
        age_model.model = load_model(model_file)

        image_files = [f for f in os.listdir(args["test_sample_dir"])]  # train_sample_dir

        for file in image_files:
            file_path = os.path.join(args["test_sample_dir"], file)  # train_sample_dir

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_dim, img_dim))

            basemodel_preprocess = age_model.preprocessing_function()
            image = basemodel_preprocess(image)
            image = np.expand_dims(image, axis=0)

            file_name = os.path.splitext(file)[0]
            true_age = int(file_name.split("_")[1])
            print(f"True label: {true_age}")

            if args["type"] == "classification":
                prediction = age_model.model.predict(image)

                mean_ind = np.where(prediction[0] == np.amax(prediction[0]))[0][0]
                print(f"Predicted label: {mean_ind}")
            else:
                prediction = age_model.model.predict(image)
                print(f"Predicted label: {prediction[0][0]*100}")


if __name__ == "__main__":
    train_model()
