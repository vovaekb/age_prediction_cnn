import os
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
from imgaug import augmenters as iaa
from facematch.age_prediction.utils.utils import build_age_vector, age_ranges_number, get_age_range_index

AGES_NUMBER = 100
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
        self.predict_gender = args["predict_gender"] if "predict_gender" in args else False
        self.range_mode = args["range_mode"] if "range_mode" in args else False
        self.age_classes_number = age_ranges_number() if self.range_mode else AGES_NUMBER
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

        self.X = self.augmentor(self.X)
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
            self.y_age = np.empty((self.batch_size, self.age_classes_number))
        else:
            self.y_age = np.empty((self.batch_size, 1))

        if self.predict_gender:
            self.y_gender = np.empty((self.batch_size, GENDERS_NUMBER))

        for i, file in enumerate(batch_samples):
            self.process_file(file, i)

    def augmentor(self, images):
        "Apply data augmentation"
        seq = iaa.Sequential(
            [iaa.Fliplr(0.5), iaa.GaussianBlur((0, 0.5))], random_order=True  # horizontally flip 50% of all images
        )
        return seq.augment_images(images)

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
        age = min(age, MAX_AGE)

        # Save AGE label and image to training dataset
        if self.model_type == "classification":
            if self.range_mode:
                range_index = get_age_range_index(age)
                # transform label to categorical vector
                self.y_age[index,] = to_categorical(range_index, self.age_classes_number)
            else:
                # Build AGE vector
                age_vector = build_age_vector(age, self.age_deviation)
                self.y_age[index,] = age_vector
        else:
            age = float(age / MAX_AGE)
            self.y_age[index] = age

        if self.predict_gender:
            gender = int(file_name.split("_")[GENDERS_NUMBER])
            # transform label to categorical vector
            self.y_gender[index] = to_categorical(gender, GENDERS_NUMBER)

    def load_sample_files(self):
        """
        Loads file names of training samples
        :return:
        """
        self.sample_files = [
            os.path.join(self.samples_directory, f)
            for f in os.listdir(self.samples_directory)
            if (f.endswith("JPG") or f.endswith("jpg"))
        ]

        self.dataset_size = len(self.sample_files)
