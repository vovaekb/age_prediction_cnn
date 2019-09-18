import importlib
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from facematch.age_prediction.utils.utils import age_ranges_number
from facematch.age_prediction.utils.metrics import earth_movers_distance, age_mae

AGES_NUMBER = 100
AGE_RANGES_NUMBER = 10
RANGE_LENGTH = 5
AGE_RANGES_UPPER_THRESH = 80
GENDERS_NUMBER = 2
MOBILENET_MODEL_NAME = "MobileNetV2"
RESNET_MODEL_NAME = "ResNet50"


class AgeClassificationNet:
    def __init__(self, base_model, img_shape, range_mode=False, predict_gender=False):
        self.base_model = base_model
        self.img_shape = img_shape
        self.range_mode = range_mode
        self.predict_gender = predict_gender
        self._get_base_module()

    def build(self):
        if self.base_model == MOBILENET_MODEL_NAME:
            base_model = MobileNetV2(input_shape=self.img_shape, include_top=False, weights="imagenet")
        elif self.base_model == RESNET_MODEL_NAME:
            base_model = ResNet50(input_shape=self.img_shape, include_top=False, weights="imagenet")

        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        if self.range_mode:
            age_classes_number = age_ranges_number()
            age_output = Dense(units=age_classes_number, activation="softmax", name="age_output")(x)
        else:
            age_output = Dense(units=AGES_NUMBER, activation="softmax", name="age_output")(x)

        if not self.predict_gender:
            self.model = Model(inputs=base_model.input, outputs=age_output)
        else:
            gender_output = Dense(units=GENDERS_NUMBER, activation="softmax", name="gender_output")(x)
            self.model = Model(inputs=base_model.input, outputs=[age_output, gender_output])

    def _get_base_module(self):
        # import Keras base model module
        if self.base_model == MOBILENET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")
        elif self.base_model == RESNET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.resnet50")

    def compile(self):
        learning_rate = 0.001
        optimizer = Adam(lr=learning_rate)

        if self.range_mode:
            age_loss = "categorical_crossentropy"
            age_metrics = ["accuracy"]
        else:
            age_loss = earth_movers_distance
            age_metrics = [age_mae, "accuracy"]

        if not self.predict_gender:
            self.model.compile(optimizer=optimizer, loss=age_loss, metrics=age_metrics)
        else:
            self.model.compile(
                optimizer=optimizer,
                loss={"age_output": age_loss, "gender_output": "categorical_crossentropy"},
                metrics={"age_output": age_metrics, "gender_output": "accuracy"},
            )

    def preprocessing_function(self):
        return self.base_module.preprocess_input
