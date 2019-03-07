import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer


class ELMoLayer(Layer):
    def __init__(self, max_len, **kwargs):
        self.dimensions = 1024
        self.max_len = max_len
        self.trainable = False
        super(ELMoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ELMoLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(tf.squeeze(tf.cast(x, tf.string)),
                           as_dict=True,
                           signature='default')['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, self.max_len * ['_PAD_'])

    def compute_output_shape(self, input_shape):
        return None, self.max_len, self.dimensions
