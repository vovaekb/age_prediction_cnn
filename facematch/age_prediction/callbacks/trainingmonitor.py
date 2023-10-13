# import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        """
        Initializes the training history dictionary and loads the history from a JSON file.

        Parameters:
            logs (dict): Optional. A dictionary to store the training logs. Default is an empty dictionary.

        Returns:
            None
        """
        # initialize the history dictionary
        self.H = defaultdict(list)  # {}

        """
        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][: self.startAt]
        """

    def on_epoch_end(self, epoch, logs={}):
        """
        Called at the end of each epoch during training.
        
        Parameters:
            epoch (int): The current epoch number.
            logs (dict): Dictionary containing metrics logged during training.
        
        Returns:
            None
        
        Description:
            This function updates the loss, accuracy, and other metrics for the entire training process.
            It appends the metrics to the corresponding lists in the 'H' dictionary.
            If the 'jsonPath' attribute is set, it serializes the training history to a JSON file.
            It also plots the training loss and accuracy using matplotlib and saves the figure to 'figPath'.
        """
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        # print(logs)
        for (k, v) in logs.items():
            self.H[k].append(v)
            # l = self.H.get(k, [])
            # l.append(v)


        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["age_output_acc"], label="age_output_acc")
            plt.plot(N, self.H["val_age_output_acc"], label="val_age_output_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.figPath)
            plt.close()
