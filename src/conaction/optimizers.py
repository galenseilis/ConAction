import tensorflow as tf

class ConstantStep(tf.keras.optimizers.Optimizer):

    def __init__(self,
                 learning_rate=0.0001,
                 name="ConstantStep",
                 **kwargs):
        super().__init__(name)
        self._lr = learning_rate

    def _create_slots(self, var_list):
        pass

    def _resource_apply_dense(self, grad, var):
        return tf.raw_ops.ResourceApplyGradientDescent(var=var.handle,
                                                       alpha=self._lr,
                                                       delta=tf.math.sign(grad)
                                                       )

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

if __name__ == '__main__':

    class Linear(tf.keras.models.Model):
        def __init__(self, units=1, input_dim=1):
            super(Linear, self).__init__()
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(input_dim, units), dtype="float32"),
                trainable=True,
            )
            b_init = tf.zeros_initializer()
            self.b = tf.Variable(
                initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
            )

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b

    train_x = tf.reshape(tf.linspace(0,10,num=10), (10,1))
    train_y = 3 * train_x + 4.0
    model = Linear()
    opt = ConstantStep(learning_rate=0.1)
    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(optimizer=opt, loss=loss)
    model.fit(x=train_x, y=train_y, epochs=100)
