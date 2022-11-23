import tensorflow as tf
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.w = self.add_weight(name='w',
            shape=[input_dim, output_dim], initializer="random_normal", trainable = True)
        self.b = self.add_weight(name='b',
            shape=[output_dim], initializer="random_normal", trainable = True)

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred

class AddNorm(tf.keras.Model):
    def __init__(self, dropout, skrm):
        super(AddNorm, self).__init__()
        self.skrms = skrm
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self, inputs):
        (X,Y) = inputs
        add = self.dropout(Y) + X 
        output = self.ln(add)
        self.skrms.Count(Y, add)
        self.skrms.Count(add, output)
        return output

class PositionWiseFFN(tf.keras.Model):
    def __init__(self, config, parameters,skrm,index):
        super(PositionWiseFFN, self).__init__()
        self.config = config 
        self.skrms = skrm
        self.parameters = parameters 
        self.index = index 
        self.dense1 = LinearLayer(config.ffnNumInput, config.ffnNumHiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = LinearLayer(config.ffnNumHiddens, config.ffnNumInput)

    def call(self, X):
        output1 = self.dense1(X)
        output2 = self.relu(output1)
        output3 = self.dense2(output2)
        self.skrms.Count(X, output1)
        self.skrms.Count(output1, output2)
        self.skrms.Count(output2, output3)
        return output3

    def LoadParameters(self):
        self.dense2.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense1.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense2.bias"]])
        self.dense1.set_weights([self.parameters[f"encoder.blks.{self.index}.ffn.dense2.weight"],self.parameters[f"encoder.blks.{self.index}.ffn.dense1.bias"]])