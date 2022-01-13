import tensorflow as tf
 
class LossHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.weights = []
        self.n = 0
        self.n += 1

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        w = self.model.get_weights()
        self.weights.append([x.flatten() for x in w])
        self.n += 1
