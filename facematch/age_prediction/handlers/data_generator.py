import os
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
from facematch.age_prediction.utils.utils import build_age_vector

CLASSES_NUMBER = 100
GENDERS_NUMBER = 2
MAX_AGE = 100


class DataGenerator(keras.utils.Sequence):
    """inherits from Keras Sequence base object"""

    def __init__(self, args, samples_directory, basemodel_preprocess, generator_type, shuffle):
        self.samples_directory = samples_directory
        self.model_type = args["type"]
        self.base_model = args["base_model"]
        self.basemodel_preprocess = basemodel_preprocess
        self.batch_size = args["batch_size"]
        self.sample_files = []
        self.img_dims = (args["img_dim"], args["img_dim"])  # dimensions that images get resized into when loaded
        self.age_deviation = args["age_deviation"]
        self.predict_gender = args['predict_gender'] if 'predict_gender' in args else False
        self.dataset_size = None
        self.generator_type = generator_type
        self.shuffle = shuffle

        self.load_sample_files()
        self.indexes = np.arange(self.dataset_size)

        self.on_epoch_end()  # for training data: call ensures that samples are shuffled in first epoch if shuffle is set to True

    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))  #  number of batches per epoch

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]  # get batch indexes
        batch_samples = [self.sample_files[i] for i in batch_indexes]

        self.__data_generator(batch_samples)
        if not self.predict_gender:
            return self.X, self.y_age
        else:
            return self.X, [self.y_age, self.y_gender]

    def on_epoch_end(self):
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generator(self, batch_samples):
        # initialize images and labels tensors for faster processing
        self.X = np.empty((self.batch_size, *self.img_dims, 3))

        if self.model_type == "classification":
            self.y_age = np.empty((self.batch_size, CLASSES_NUMBER))
        else:
            self.y_age = np.empty((self.batch_size, 1))

        if self.predict_gender:
            self.y_gender = np.empty((self.batch_size, GENDERS_NUMBER))

        for i, file in enumerate(batch_samples):
            self.process_file(file, i)

    def process_file(self, file, index):
        # Load image
        file_path = os.path.join(self.samples_directory, file)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, self.img_dims)

        # apply basenet specific preprocessing
        image = self.basemodel_preprocess(image)

        image = np.expand_dims(image, axis=0)
        self.X[index,] = image

        # Obtain age
        file_name = os.path.splitext(file)[0]
        age = int(file_name.split("_")[1])

        # Save AGE label and image to training dataset
        if self.model_type == "classification":
            # Build AGE vector
            age_vector = build_age_vector(age, self.age_deviation)
            self.y_age[index,] = age_vector
        else:
            age = float(age / 100.0)
            self.y_age[index] = age

        if self.predict_gender:
            gender = int(file_name.split("_")[2])
            # TODO: apply label encoding (0 -> [1, 0])
            self.y_gender[index] = to_categorical(gender, 2)

    def load_sample_files(self):
        """
        Processes batch of samples sending API requests with every sample one by one
        :return:
        """
        self.sample_files = [
            os.path.join(self.samples_directory, f)
            for f in os.listdir(self.samples_directory)
            if (f.endswith("JPG") or f.endswith("jpg"))
        ]

        self.dataset_size = len(self.sample_files)
