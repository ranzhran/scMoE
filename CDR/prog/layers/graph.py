import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, constraints, activations

class GraphLayer(layers.Layer):
    def __init__(self, step_num=1, activation=None, **kwargs):
        """Initialize the layer.

        :param step_num: Number of steps to consider for the connectivity graph.
        :param activation: Activation function after convolution.
        :param kwargs: Additional arguments for parent class.
        """
        super(GraphLayer, self).__init__(**kwargs)
        self.step_num = step_num
        self.activation = activations.get(activation)
        self.supports_masking = True

    def get_config(self):
        config = {
            'step_num': self.step_num,
            'activation': activations.serialize(self.activation),
        }
        base_config = super(GraphLayer, self).get_config()
        return {**base_config, **config}

    def _get_walked_edges(self, edges, step_num):
        """Get the connection graph within `step_num` steps.

        :param edges: The graph in a single step.
        :param step_num: Number of steps.
        :return: The new graph that has the same shape as `edges`.
        """
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(tf.linalg.matmul(edges, edges), step_num // 2)
        if step_num % 2 == 1:
            deeper += edges
        return tf.cast(tf.greater(deeper, 0.0), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        features, edges = inputs
        edges = tf.cast(edges, dtype=tf.float32)
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = self.activation(self._call(features, edges))
        return outputs

    def _call(self, features, edges):
        raise NotImplementedError('The class is not intended to be used directly.')

class GraphConv(GraphLayer):
    """Graph convolutional layer."""

    def __init__(self, units, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 kernel_constraint=None, use_bias=True, bias_initializer='zeros',
                 bias_regularizer=None, bias_constraint=None, **kwargs):
        """Initialize the layer.

        :param units: Number of output units.
        :param kernel_initializer: Initializer for the kernel weight matrix.
        :param kernel_regularizer: Regularizer for the kernel weight matrix.
        :param kernel_constraint: Constraint for the kernel weight matrix.
        :param use_bias: Whether to use bias term.
        :param bias_initializer: Initializer for the bias vector.
        :param bias_regularizer: Regularizer for the bias vector.
        :param bias_constraint: Constraint for the bias vector.
        :param kwargs: Additional arguments for parent class.
        """
        super(GraphConv, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W'.format(self.name),
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b'.format(self.name),
            )
        super(GraphConv, self).build(input_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(GraphConv, self).get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.units,)

    def compute_mask(self, inputs, mask=None):
        return mask[0] if mask is not None and mask[0] is not None else None

    def _call(self, features, edges):
        features = tf.linalg.matmul(features, self.W)
        if self.use_bias:
            features += self.b
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        return tf.linalg.matmul(tf.transpose(edges, perm=[0, 2, 1]), features)

class GraphPool(GraphLayer):
    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return mask[0] if mask is not None and mask[0] is not None else None

class GraphMaxPool(GraphPool):
    NEG_INF = -1e38

    def _call(self, features, edges):
        node_num = tf.shape(features)[1]
        features = tf.tile(tf.expand_dims(features, axis=1), [1, node_num, 1, 1])
        return tf.reduce_max(features + tf.expand_dims((1.0 - edges) * self.NEG_INF, axis=-1), axis=2)

class GraphAveragePool(GraphPool):
    def _call(self, features, edges):
        return tf.linalg.matmul(tf.transpose(edges, perm=[0, 2, 1]), features) \
               / (tf.reduce_sum(edges, axis=2, keepdims=True) + tf.keras.backend.epsilon())
