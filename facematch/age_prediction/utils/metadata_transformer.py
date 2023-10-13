import os
import shutil


class MetadataTransformer:
    def __init__(self, args):
        """
        Initializes a new instance of the class.
        
        Args:
            args (dict): A dictionary containing the arguments.
                - sample_dir (str): The directory path of the samples.
                - output_directory (str): The directory path for the output.
        """
        self.samples_directory = args["sample_dir"]
        self.output_directory = args["output_directory"]

    def process_file(self, file, index):
        """
        Reads age, gender and image Id from training file and copies it to the output directory.

        Parameters:
            file (str): The name of the file to be processed.
            index (int): The index of the file in the list of files.

        Returns:
            None
        """
        print(f"Preprocessing file {file}")

        # Load image
        file_path = os.path.join(self.samples_directory, file)

        # Read age, gender from file name
        age = int(file.split("_")[0])
        gender = int(file.split("_")[1])
        image_id = file.split(".")[0].split("_")[-1]

        new_file_name = f"{image_id}_{gender}_{age}.jpg"
        new_file_path = os.path.join(self.output_directory, new_file_name)
        shutil.copy(file_path, new_file_path)

    def process_samples_directory(self):
        """
        Processes batch of samples sending API requests with every sample one by one
        :return:
        """
        print("Processing samples directory...")

        self.image_files = [
            f
            for f in os.listdir(self.samples_directory)
            if (f.endswith("JPG") or f.endswith("jpg"))
            and os.path.getsize(os.path.join(self.samples_directory, f)) > 500
        ]
        # print(self.image_files)

        for i, file in enumerate(self.image_files):
            self.process_file(file, i)

        print("Preprocessing samples directory complete")
