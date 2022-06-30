import tensorflow as tf
import numpy as np

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

class GradientDataSet():
    '''
    Use gradient descent to construct
    a dataset with a given correlation
    structure.
    '''

    def __init__(self):
        pass
##
if __name__ == '__main__':
    learning_rate = 10**-2
    tol = 0.01
    loss = np.inf
    epoch = 0

    w = tf.Variable(np.random.normal(0, 1, 3000).reshape(1000,3))
    prev_w = w + tol * 10

    # 'Better' break condition: tf.math.reduce_sum(tf.abs(w - prev_w))
    loss_history = []
    while loss > tol:
        np.savetxt(f'/home/galen/Projects/3corr_{str(epoch).zfill(6)}.csv', w.numpy(), delimiter=",")
        epoch += 1
        w = tf.Variable(w)
        with tf.GradientTape() as tape:
            loss = tf.math.abs(1-tfcorr(w)) #+ tf.math.sqrt(tf.math.reduce_sum(tf.square(w)))
            print(epoch, loss)
            loss_history.append(loss)
            
        grad = tape.gradient(loss, w)
        prev_w = w
        w = w - learning_rate * grad

    import matplotlib.pyplot as plt
    plt.plot(range(1, epoch+1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel(r'$|1 - R[X,Y,Z]|$')
    plt.tight_layout()
    plt.savefig('loss_history_pos.pdf')
    plt.close()

    np.savetxt(f'/home/galen/Projects/3corr_{str(epoch).zfill(6)}.csv', w.numpy(), delimiter=",")
