import tensorflow as tf

pred = tf.constant([[0.1, 0.4, 0.5], [0.2, 0.1, 0.7]])
actual = tf.constant([[0.2, 0.3, 0.5], [0.1, 0.3, 0.6]])

tf.nn.softmax_cross_entropy_with_logits()