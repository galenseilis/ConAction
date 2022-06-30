import tensorflow as tf
from itertools import combinations

class ProductDeviations(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()
        self.name = ""

    def call(self, y_true, y_pred):

        result = y_pred - tf.math.reduce_mean(y_pred, axis=0)
        result = tf.linalg.matmul(result, result, transpose_a=True)
        result = result - tf.linalg.band_part(result, 0, 0)
        result = tf.math.pow(result, 2)
        result = tf.math.reduce_mean(result)

        return - result

