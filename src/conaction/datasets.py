import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def tfcov(X):
    num = X - tf.math.reduce_mean(X, axis=0)
    num = tf.math.reduce_prod(num, axis=1)
    num = tf.math.reduce_mean(num)
    return num

def tfcorr(X):
    num = X - tf.math.reduce_mean(X, axis=0)
    num = tf.math.reduce_prod(num, axis=1)
    num = tf.math.reduce_mean(num)
    denom = X - tf.math.reduce_mean(X, axis=0)
    denom = tf.abs(denom)
    denom = denom ** X.shape[1]
    denom = tf.reduce_mean(denom, axis=0)
    denom = denom ** (1 / X.shape[1])
    denom = tf.reduce_prod(denom)
    return num / denom

def tfproddev(X):
    return tf.reduce_prod(X - tf.reduce_mean(X, axis=0), axis=1)

def tfexcesskurt(X):
    res = X - tf.reduce_mean(X)
    num = tf.reduce_mean(res ** 4)
    denom = tf.reduce_mean(res ** 2) ** 2
    return num / denom - 3

def tfskew(X):
    res = X - tf.reduce_mean(X, axis=0)
    num = tf.reduce_mean(res ** 3)
    denom = tf.reduce_mean(res ** 2) ** (3 / 2)
    return denom
##
if __name__ == '__main__':
    learning_rate = 10**-2
    tol = 0.01
    loss = np.inf
    epoch = 0

    w = tf.Variable(np.random.normal(0,1, 4000).reshape(1000,4))
    prev_w = w + tol * 10

    # 'Better' break condition: tf.math.reduce_sum(tf.abs(w - prev_w))

    while loss > tol:
        epoch += 1
        w = tf.Variable(w)
        with tf.GradientTape() as tape:
            loss = tf.abs(-1 - tfcorr(w)) + tf.math.sqrt(tf.math.reduce_sum(tf.square(w)))
            print(epoch, loss)
            
        grad = tape.gradient(loss, w)
        prev_w = w
        w = w - learning_rate * grad
