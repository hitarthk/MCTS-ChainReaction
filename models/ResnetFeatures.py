import tensorflow as tf
from tensorflow.keras import *

class ResnetBlock(tf.keras.Model):
    def __init__(self, filters, conv_size, l2Weight, name, **kwargs):
        super(ResnetBlock, self).__init__(name, **kwargs)
        self.filters = filters
        self.conv_size = conv_size

        self.conv2a = tf.keras.layers.Conv2D(filters, conv_size, padding='same', kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters, conv_size, padding='same', kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.bn2b = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class IntuitionPolicy(tf.keras.Model):
    def __init__(self, totalRows, totalCols, numPlayers, l2Weight = 0.0001, numResnetBlocks = 10, filters = 64):
        super(IntuitionPolicy, self).__init__()
        self.totalRows = totalRows
        self.totalCols = totalCols
        self.numPlayers = numPlayers
        self.numResnetBlocks = numResnetBlocks

        self.conv = tf.keras.layers.Conv2D(filters, (3, 3), padding ='same', kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.bn = tf.keras.layers.BatchNormalization()

        self.resnetBlocks = []

        for i in range(numResnetBlocks):
            self.resnetBlocks.append(ResnetBlock(filters, 3, l2Weight, 'residual_block_'+str(i)))

        self.convPolicy = tf.keras.layers.Conv2D(2, (1, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.bnPolicy = tf.keras.layers.BatchNormalization()
        self.fcPolicy = tf.keras.layers.Dense(totalRows * totalCols, kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.policyFlatten = tf.keras.layers.Flatten()

        self.convValue = tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.bnValue = tf.keras.layers.BatchNormalization()

        self.valueFlatten = tf.keras.layers.Flatten()

        self.fcValue1 = tf.keras.layers.Dense(128, kernel_regularizer = tf.keras.regularizers.l2(l2Weight))
        self.fcValue2 = tf.keras.layers.Dense(1, kernel_regularizer = tf.keras.regularizers.l2(l2Weight))

    def call(self, input, training = False):
        x = self.conv(input)
        x = self.bn(x)
        x = tf.nn.relu(x)

        for i in range(self.numResnetBlocks):
            x = self.resnetBlocks[i](x, training)

        policyFeatures = self.convPolicy(x)
        policyFeatures = self.bnPolicy(policyFeatures, training)
        policyFeatures = tf.nn.relu(policyFeatures)
        policyFeatures = self.policyFlatten(policyFeatures)
        actionLogitProbs = self.fcPolicy(policyFeatures)
        actionProbs = tf.nn.softmax(actionLogitProbs)

        valueFeatures = self.convValue(x)
        valueFeatures = self.bnValue(valueFeatures)
        valueFeatures = self.valueFlatten(valueFeatures)
        valueFeatures = self.fcValue1(valueFeatures)
        valueFeatures = tf.nn.relu(valueFeatures)
        valueFeatures = self.fcValue2(valueFeatures)
        valueFeatures = tf.squeeze(valueFeatures)
        valueFeatures = tf.nn.tanh(valueFeatures)

        return tf.tuple([actionProbs, valueFeatures])

if __name__=='__main__':
    intuitionPolicy = IntuitionPolicy(3, 3, 2, 5, 2)
    x = tf.convert_to_tensor
    input_tensor = tf.constant([[[[1, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [0, 1]], [[2, 0], [0, 2], [2, 0]]], [[[1, 0], [0, 1], [0, 0]], [[0, 0], [1, 0], [0, 1]], [[2, 0], [0, 2], [2, 0]]]], dtype = tf.float32)
    out = intuitionPolicy(input_tensor)
    print(intuitionPolicy.summary())
    print(f'out: {out}')

    logits = tf.constant([[-1, 1], [-2, 2]], dtype = tf.float32)
    probs = tf.nn.softmax(logits)
    print(probs)