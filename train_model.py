import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from facematch.age_prediction.handlers.data_generator import DataGenerator
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from facematch.age_prediction.optimization.clr_callback import CyclicLR
from facematch.age_prediction.optimization.learningratefinder import LearningRateFinder
from facematch.age_prediction.callbacks.trainingmonitor import TrainingMonitor
from facematch.age_prediction.models.age_classification_net import AgeClassificationNet
from facematch.age_prediction.models.age_regression_net import AgeRegressionNet
from facematch.age_prediction.utils.metrics import earth_movers_distance, age_mae
from facematch.age_prediction.utils.utils import get_range

EPOCHS = 15
DATASET_SIZE = 1000

matplotlib.use("Agg")

parser = argparse.ArgumentParser()

get_custom_objects().update({"earth_movers_distance": earth_movers_distance, "age_mae": age_mae})


def train_model():
    parser.add_argument("-d", "--train_sample_dir", help="Path to raw images for training")
    parser.add_argument("-v", "--test_sample_dir", help="Path to images for validation")
    parser.add_argument("-w", "--model_path", help="Path to model JSON and weights")
    parser.add_argument("-s", "--img_dim", type=int, help="Dimension of input images for training (width, height)")
    parser.add_argument("-bs", "--batch_size", type=int, default=5, help="Size of batch to use for training")
    parser.add_argument(
        "-o",
        "--lr_scheduler",
        type=str,
        default="reduce_lr_on_plateau",
        help="Learning rate scheduler to use (reduce_lr_on_plateau, cyclic_lr)",
    )
    parser.add_argument("-dev", "--age_deviation", type=int, default=5, help="Deviation in age vector")
    parser.add_argument("-t", "--type", type=str, help="Type of model to use (regression, classification)")  #
    parser.add_argument(
        "-rm",
        "--range_mode",
        default=False,
        type=bool,
        help="Run age prediction in range mode (age prediction in ranges like 0 - 5, 6 - 10 etc). If not set run in age vector mode (normal probability histogram)",
    )
    parser.add_argument(
        "-b", "--base_model", type=str, help="Base model to use in the NN model (MobileNetV2, ResNet50)"
    )
    parser.add_argument("-l", "--load", default=False, type=bool, help="Load model from file")
    parser.add_argument("-gnd", "--predict_gender", default=False, type=bool, help="Apply gender prediction")
    parser.add_argument("-ft", "--fine_tuning", default=False, type=bool, help="Apply fine tuning to model")
    parser.add_argument(
        "-f", "--lr_find", type=bool, default=False, help="whether or not to find optimal learning rate"
    )
    args = vars(parser.parse_args())

    img_dim = args["img_dim"]

    print("Initializing CNN model ...")

    if args["type"] == "classification":
        age_model = AgeClassificationNet(
            args["base_model"], (img_dim, img_dim, 3), args["range_mode"], args["predict_gender"]
        )
    else:
        # Regression model
        age_model = AgeRegressionNet(args["base_model"], (img_dim, img_dim, 3), args["predict_gender"])

    if not args["load"]:
        age_model.build()
        # age_model.model.summary()

        train_generator = DataGenerator(
            args,
            samples_directory=args["train_sample_dir"],
            generator_type="train",
            basemodel_preprocess=age_model.preprocessing_function(),
            shuffle=True,
        )
        validation_generator = DataGenerator(
            args,
            samples_directory=args["test_sample_dir"],
            generator_type="test",
            basemodel_preprocess=age_model.preprocessing_function(),
            shuffle=False,
        )

        # freeze the base model
        for layer in age_model.base_model.layers:
            layer.trainable = False

        # Compile model
        age_model.compile()

        # Fit generator and start training
        print("Starting training ...")

        # Automatic finding best learning rate
        if args["lr_find"]:
            # define the path to the output learning rate finder plot, training
            # history plot and cyclical learning rate plot
            LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
            # TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
            # CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])

            lrf = LearningRateFinder(age_model.model)
            lrf.find(
                train_generator,
                1e-10,
                1e1,
                stepsPerEpoch=np.ceil((DATASET_SIZE / float(args["batch_size"]))),
                batchSize=args["batch_size"],
                epochs=3,
            )

            # plot the loss for the various learning rates and save the
            # resulting plot to disk
            lrf.plot_loss()
            plt.savefig(LRFIND_PLOT_PATH)

            # gracefully exit the script so we can adjust our learning rates
            # in the config and then train the network for our full set of
            # epochs
            print("[INFO] learning rate finder complete")
            print("[INFO] examine plot and adjust learning rates before training")
            return

        # Add model checkpoint
        checkpoint = ModelCheckpoint("model_out.hdf5", monitor="val_loss", verbose=1, save_best_only=True)

        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)

        # Add training monitor
        # construct the set of callbacks
        figPath = os.path.sep.join(["output", "{}.png".format(os.getpid())])
        jsonPath = os.path.sep.join(["output", "{}.json".format(os.getpid())])

        training_monitor = TrainingMonitor(figPath, jsonPath=jsonPath)

        # Apply learning rate schedules
        if args["lr_scheduler"] == "reduce_lr_on_plateau":
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
            )  # patience=5, min_lr=0.00001
        else:
            # We using the triangular learning rate policy and
            #  base_lr (initial learning rate which is the lower boundary in the cycle)
            lr_scheduler = CyclicLR(
                mode="triangular",
                base_lr=1e-4,
                max_lr=1e-1,
                step_size=8 * (DATASET_SIZE / args["batch_size"]),  # base_lr=0.0001, max_lr=0.01
            )

        # add the learning rate schedule to the list of callbacks
        callbacks = [checkpoint, training_monitor, lr_scheduler]

        # Applying fine tuning
        if args["fine_tuning"]:
            # Train without loop and prediction monitoring
            age_model.model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                epochs=10,
                callbacks=callbacks,
                verbose=1,
            )

            # unfreeze the final set of CONV layers and make them trainable
            for layer in age_model.base_model.layers[147:]:  # 147 - MobileNetV2, 171 - ResNet50
                layer.trainable = True

            # Compile model
            age_model.compile()

            age_model.model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                epochs=35,  # 15,
                callbacks=callbacks,
                verbose=1,
            )
        else:
            age_model.model.fit_generator(
                generator=train_generator,
                validation_data=validation_generator,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1,
            )

        # save the entire model
        model_file = os.path.join(args["model_path"], "model.h5")
        age_model.model.save(model_file)
    else:
        # Load model from file
        model_file = os.path.join(args["model_path"], "model.h5")
        age_model.model = load_model(model_file)

        image_files = [f for f in os.listdir(args["test_sample_dir"])]

        for file in image_files:
            file_path = os.path.join(args["test_sample_dir"], file)
            print(f"\nPredicting for image {file}")

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_dim, img_dim))

            basemodel_preprocess = age_model.preprocessing_function()
            image = basemodel_preprocess(image)
            image = np.expand_dims(image, axis=0)

            file_name = os.path.splitext(file)[0]
            true_age = int(file_name.split("_")[1])

            prediction = age_model.model.predict(image)
            if args["type"] == "classification":
                if args["range_mode"]:
                    predict_age_index = np.where(prediction[0] == np.amax(prediction[0]))[0][0]
                    # transform range index to range
                    (range_start, range_end) = get_range(predict_age_index)
                    if range_end is None:
                        predicted_range = f"{range_start}+"
                    else:
                        predicted_range = f"({range_start}, {range_end})"
                    print(f"Predicted age range: {predicted_range}, true age: {true_age}")
                else:
                    mean_ind = np.where(prediction[0] == np.amax(prediction[0]))[0][0]
                    print(f"Predicted age: {mean_ind}, true age: {true_age}")
            else:
                # Regression mode
                if not args["predict_gender"]:
                    print(f"Predicted age: {prediction[0][0]*100}, true age: {true_age}")
                else:
                    print(f"Predicted age: {prediction[0][0][0]*100}, true age: {true_age}")

            if "predict_gender" in args and args["predict_gender"]:
                true_gender_index = int(file_name.split("_")[2])
                true_gender = "M" if true_gender_index == 0 else "F"

                max_ind = np.where(prediction[1][0] == np.amax(prediction[1][0]))[0][0]
                gender = "M" if max_ind == 0 else "F"
                print(f"Predicted gender: {gender}, true gender: {true_gender}")


if __name__ == "__main__":
    train_model()
