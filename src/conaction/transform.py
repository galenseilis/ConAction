import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

class HypersphericalRotation(tf.keras.Model):
    '''
    Trainable hyperspherical rotation of a data set.

    
    '''

    def __init__(self, units=1, input_dim=1):
        super().__init__()

    #Consider using @property
    def set_angles(self, x):
        '''
        Computes hyperspherical angles of data points and sets a trainable
        angle in phase space.

        Parameters
        ----------
        x : tf.tensor[tf.float64]
            m x n data matrix

        Returns
        -------
        theta : tf.tensor
            Hyperspherical angles of data points.
        '''
        self.tau = self.add_weight(shape=(x.shape[1]-1,), initializer="uniform", trainable=True)
        def angle(x):
            theta = []
            for j in range(len(x)-1):
                case0 = tf.math.reduce_all(tf.math.equal(x[j+1:], 0.0))
                if case0:
                    theta.append(0.0)
                elif not tf.equal(j+1, len(x)-1):
                    theta.append(tf.math.acos(x[j] / tf.linalg.norm(x[j:])).numpy())
                else:
                    if tf.math.greater_equal(x[j], 0.0):
                        theta.append(tf.math.acos(x[j] / tf.linalg.norm(x[j:])).numpy())
                    else:
                        theta.append(2 * np.pi - tf.math.acos(x[j] / tf.linalg.norm(x[j:])).numpy())
            return tf.constant(theta, dtype=tf.float64)
        self.theta = tf.map_fn(angle, x) # Test if this line is still needed.
        
        
    def call(self, inputs):
        self.r = tf.norm(inputs, axis=1)

        y = []
        shifted = self.theta + self.tau

        for j in range(shifted.shape[1]+1):
            if j == 0:
                y.append(tf.reshape(
                    self.r * tf.cos(shifted[:,j]),
                    (-1,1)
                    ))
            elif j + 1 != shifted.shape[1]+1:
                y.append(tf.reshape(
                            self.r * tf.reduce_prod(tf.sin(shifted[:,:j]), axis=1) * tf.cos(shifted[:,j]),
                            (-1,1)
                            )
                         )
            else:
                y.append(tf.reshape(
                            self.r * tf.reduce_prod(tf.sin(shifted), axis=1),
                            (-1,1)
                            )
                         )
        return tf.concat(y, axis=1)

class SingularValueShift(tf.keras.Model):
    '''
    Trainable shift in the singular values of a matrix.
    '''
    
    def __init__(self):
        super().__init__()

    def pretrain(self, inputs):
        '''
        Find the singular value decomposition of a
        matrix.

        Parameters
        ----------
        X : tf.tensor
            m x n data matrix

        Returns
        -------
        : None
        '''
        s,u,v = tf.linalg.svd(inputs)
        self.s = s
        self.u = u
        self.v = v
        self.tau = tf.Variable(tf.random.uniform(s.shape, dtype=tf.float64), trainable=True)
    
    def call(self, inputs):
        s,u,v = tf.linalg.svd(inputs)
        self.s = s
        self.u = u
        self.v = v
        result = self.s + self.tau
        result = tf.linalg.diag(result)
        result = tf.matmul(self.u, result)
        result = tf.matmul(result, self.v, transpose_b=True)
        return result

class SVDProjection(tf.keras.Model):
    '''
    Projects data along singular vectors.
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X):
        '''
        Find the singular value decomposition of a matrix,
        and then project the given data onto the singular
        vectors.

        Parameters
        ----------
        X : tf.tensor
            m x n data matrix

        Returns
        -------
        : None
        '''
        s, U, V = tf.linalg.svd(X)
        self.s = s
        self.total_s = tf.math.reduce_sum(self.s)
        self.normalized_s = self.s / self.total_s
        self.S = tf.linalg.diag(s)
        self.U = U
        self.V = V
        self.scores = tf.matmul(U, self.S)

    def call(self, X):
        '''
        Project a data matrix along singular vectors. The
        data need not be the same as used to find the singular
        value decomposition in the first place.

        Parameters
        ----------
        X : tf.tensor
            m x n data matrix

        Returns
        -------
         : tf.tensor
            Projected data.
        '''
        if hasattr(self, 'V'):
            return tf.matmul(X, self.V)
        else:
            self.fit(X)
            return tf.matmul(X, self.V)
