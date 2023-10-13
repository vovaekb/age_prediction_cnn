import importlib
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.resnet50 import ResNet50
from facematch.age_prediction.utils.utils import age_ranges_number
from facematch.age_prediction.utils.metrics import (
    earth_movers_distance,
    age_mae
)

AGES_NUMBER = 100
AGE_RANGES_NUMBER = 10
RANGE_LENGTH = 5
AGE_RANGES_UPPER_THRESH = 80
GENDERS_NUMBER = 2
MOBILENET_MODEL_NAME = "MobileNetV2"
RESNET_MODEL_NAME = "ResNet50"


class AgeClassificationNet:
    def __init__(self, base_model_name, img_shape,
                 range_mode=False, 
                 predict_gender=False):
        self.base_model_name = base_model_name
        self.img_shape = img_shape
        self.range_mode = range_mode
        self.predict_gender = predict_gender
        self._get_base_module()

    def build(self):
        """
        Builds the model by creating the base model and adding additional layers for age and gender prediction.

        Parameters:
            None

        Returns:
            None
        """
        if self.base_model_name == MOBILENET_MODEL_NAME:
            self.base_model = MobileNetV2(
                input_shape=self.img_shape, 
                include_top=False, 
                weights="imagenet"
            )
        elif self.base_model_name == RESNET_MODEL_NAME:
            self.base_model = ResNet50(
                input_shape=self.img_shape, 
                include_top=False, 
                weights="imagenet"
            )

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        if self.range_mode:
            age_classes_number = age_ranges_number()
            age_output = Dense(units=age_classes_number, activation="softmax", name="age_output")(x)
        else:
            age_output = Dense(units=AGES_NUMBER, activation="softmax", name="age_output")(x)

        if not self.predict_gender:
            self.model = Model(inputs=self.base_model.input, outputs=age_output)
        else:
            gender_output = Dense(units=GENDERS_NUMBER, activation="softmax", name="gender_output")(x)
            self.model = Model(inputs=self.base_model.input, outputs=[age_output, gender_output])

        # list indices of base model layers
        # for i, layer in enumerate(self.base_model.layers):
        #     print("{} {}".format(i, layer.__class__.__name__))

    def _get_base_module(self):
        """
        Retrieves the base module for the given base model name.

        Parameters:
            self (ClassName): An instance of the ClassName class.
        
        Returns:
            None
        """
        # import Keras base model module
        if self.base_model_name == MOBILENET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.mobilenet_v2")
        elif self.base_model_name == RESNET_MODEL_NAME:
            self.base_module = importlib.import_module("keras.applications.resnet50")

    def compile(self):
        """
        Compile the model with the specified learning rate and optimizer.

        Parameters:
            None

        Returns:
            None
        """
        learning_rate = 1e-2  # 1e-5 # 1e-1 # MIN_LR in Learning rate Finder # 0.0001 # 0.001
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
                loss_weights={"age_output": 1.0, "gender_output": 1.0},
                metrics={"age_output": age_metrics, "gender_output": "accuracy"},
            )

    def preprocessing_function(self):
<<<<<<<<<<<<<  ✨ Codeium AI Suggestion  >>>>>>>>>>>>>>
+        """
+        Preprocessing function that performs input preprocessing using the base module.
+
+        :param self: The instance of the class.
+        :return: The preprocessed input.
+        """
<<<<<  bot-54288452-b0b5-485c-8f73-5a8d87f40358  >>>>>
        return self.base_module.preprocess_input
