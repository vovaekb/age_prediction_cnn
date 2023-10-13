import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


class TerminateOnNan(Callback):
    """
    Class representing callback that terminates training when a NaN loss is encountered.
    """
    def __init__(self):
        """
        Initializes an instance of the TerminateOnNan class.
        """
        super(TerminateOnNan, self).__init__()

    def on_batch_end(self, batch, logs=None):
        """
        Function called at the end of every batch. It prints a message and terminates training if the loss is invalid.

        Parameters:
            batch (int): The batch number.
            logs (dict, optional): Dictionary of logs. Defaults to None.

        Returns:
            None
        """
        logs = logs or {}
        loss = logs.get("loss")

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print("Batch %d: Invalid loss, terminating training" % (batch))
                self.model.stop_training = True
