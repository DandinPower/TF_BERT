import tensorflow as tf
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.w = self.add_weight(name='w',
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_weight(name='b',
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        y_pred = self.kernel.bits_quant(tf.matmul(inputs, self.w)) + self.kernel.bits_quant(self.b)
        return self.kernel.bits_quant(y_pred)

class AddNorm(tf.keras.Model):
    def __init__(self, dropout):
        super(AddNorm, self).__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        return self.kernel.bits_quant(self.ln(self.dropout(Y) + X))

class PositionWiseFFN(tf.keras.Model):
    def __init__(self, config, parameters,index):
        super(PositionWiseFFN, self).__init__()
        self.kernel = tf.load_op_library('./models/operations/bits_quant.so')
        self.config = config 
        self.parameters = parameters 
        self.index = index 
        self.dense1 = LinearLayer(config.ffnNumInput, config.ffnNumHiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = LinearLayer(config.ffnNumHiddens, config.ffnNumInput)

    def call(self, X):
        return self.dense2(self.kernel.bits_quant(self.relu(self.dense1(X))))

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])