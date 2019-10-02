import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


class TerminateOnNan(Callback):
    def __init__(self):
        super(TerminateOnNan, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print("Batch %d: Invalid loss, terminating training" % (batch))
                self.model.stop_training = True
