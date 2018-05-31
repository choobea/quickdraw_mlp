import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops.nn_ops import leaky_relu

from utils.network_summary import count_parameters

from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

class VGGClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, filter_size=[3, 3]):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.filter_size = filter_size

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('VGGNet'):
                outputs = image_input
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if (j == self.inner_layer_depth-1) and self.strided_dim_reduction:
                                    stride = 2
                                else:
                                    stride = 1
                                outputs = tf.layers.conv2d(outputs, self.layer_stage_sizes[i], self.filter_size,
                                                           strides=(stride, stride),
                                                           padding='SAME', activation=None)
                                outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                                layer_features.append(outputs)
                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        if self.strided_dim_reduction==False:
                            outputs = tf.layers.max_pooling2d(outputs, pool_size=(2, 2), strides=2)

                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                                                                              # apply dropout only at dimensionality
                                                                              # reducing steps, i.e. the last layer in
                                                                              # every group

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return c_conv_encoder, layer_features


class FCCLayerClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True):
        """
        Initializes a Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('FCCLayerNet'):
                outputs = image_input
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                outputs = tf.layers.dense(outputs, units=self.layer_stage_sizes[i])
                                outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                                layer_features.append(outputs)
                                if self.batch_norm_use:
                                    outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                                                                              # apply dropout only at dimensionality
                                                                              # reducing steps, i.e. the last layer in
                                                                              # every group

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "FCCLayerNet")

        return c_conv_encoder, layer_features


class RNNLayerClassifier:
    def __init__(self, batch_size, layer_stage_sizes, rnn_cell_type, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, bidirectional=False, conv_rnn_sizes=[]):

        """
        Initializes a RNN Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.rnn_cell_type = rnn_cell_type
        self.bidirectional = bidirectional
        self.conv_rnn_sizes = conv_rnn_sizes


    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the network producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        used = tf.sign(tf.reduce_max(tf.abs(image_input), 2))
        stroke_lengths = tf.cast(tf.reduce_sum(used, 1), tf.int32)
        # num_rnn_nodes = 128
        # num_rnn_layers = 3

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('RNNLayerNet'):
                outputs = image_input
                if self.bidirectional:
                    if self.rnn_cell_type == 'BasicRNNCell':
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell
                    elif self.rnn_cell_type == 'LSTMCell':
                        rnn_cell = tf.nn.rnn_cell.LSTMCell
                    elif self.rnn_cell_type == 'GRUCell':
                        rnn_cell = tf.nn.rnn_cell.GRUCell
                    else:
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell

                    for i in range(len(self.conv_rnn_sizes)):
                        with tf.variable_scope('conv_stage_{}'.format(i)):
                            if self.batch_norm_use:
                                outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                             center=True, is_training=training, renorm=False)
                            # Add dropout layer if enabled and not first convolution layer.
                            if i > 0:
                                outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                            outputs = tf.layers.conv1d(outputs,
                                                       filters=self.conv_rnn_sizes[i],
                                                       kernel_size=3,
                                                       activation=None,
                                                       strides=1,
                                                       padding='SAME',
                                                       name='conv1d_%d' % i)
                            layer_features.append(outputs)

                    with tf.variable_scope('rnn_stage'):
                        # apply dropout
                        # cells_fw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.AttentionCellWrapper(rnn_cell(self.layer_stage_sizes[i]), 10), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.layer_stage_sizes))]
                        cells_fw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.layer_stage_sizes[i]), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.layer_stage_sizes))]
                        # cells_bw = [tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.AttentionCellWrapper(rnn_cell(self.layer_stage_sizes[i]), 10), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.layer_stage_sizes))]
                        cells_bw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.layer_stage_sizes[i]), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.layer_stage_sizes))]
                        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=cells_fw,
                                cells_bw=cells_bw,
                                inputs=outputs,
                                sequence_length=stroke_lengths,
                                dtype=tf.float32,
                                scope="rnn_classification")
                        layer_features.append(outputs)
                        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
                        # the number of nodes in the last layer.
                        # do also experiment where we do not do these additional things
                        mask = tf.tile(
                            tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(outputs)[1]), 2),
                            [1, 1, tf.shape(outputs)[2]])
                        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
                        outputs = tf.reduce_sum(zero_outside, axis=1)
                        layer_features.append(outputs)
                else:
                    for i in range(len(self.layer_stage_sizes)):
                        with tf.variable_scope('RNN_stage_{}'.format(i)):
                            if self.rnn_cell_type == 'BasicRNNCell':
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.layer_stage_sizes[i])
                            elif self.rnn_cell_type == 'LSTMCell':
                                rnn_cell = tf.nn.rnn_cell.LSTMCell(self.layer_stage_sizes[i])
                            elif self.rnn_cell_type == 'GRUCell':
                                rnn_cell = tf.nn.rnn_cell.GRUCell(self.layer_stage_sizes[i])
                            else:
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.layer_stage_sizes[i])

                            # this is where dropout is applied in rnn tutorials

                            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=1.0 - dropout_rate)

                            outputs, state = tf.nn.dynamic_rnn(rnn_cell, outputs,
                                                               sequence_length=stroke_lengths,
                                                               dtype=tf.float32)

                            layer_features.append(outputs)

                # for i in range(len(self.layer_stage_sizes)):
                #     with tf.variable_scope('conv_stage_{}'.format(i)):
                #         if self.batch_norm_use:
                #             outputs = batch_norm(outputs, decay=0.99, scale=True,
                #                                          center=True, is_training=training, renorm=False)
                #         # Add dropout layer if enabled and not first convolution layer.
                #         if i > 0:
                #             outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                #         outputs = tf.layers.conv1d(outputs,
                #                                    filters=self.layer_stage_sizes[i],
                #                                    kernel_size=3,
                #                                    activation=None,
                #                                    strides=1,
                #                                    padding='SAME',
                #                                    name='conv1d_%d' % i)
                #         layer_features.append(outputs)
                #
                # with tf.variable_scope('rnn_stage'):
                #     cell = tf.nn.rnn_cell.BasicLSTMCell
                #     cells_fw = [cell(num_rnn_nodes) for _ in range(num_rnn_layers)]
                #     cells_bw = [cell(num_rnn_nodes) for _ in range(num_rnn_layers)]
                #     outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                #             cells_fw=cells_fw,
                #             cells_bw=cells_bw,
                #             inputs=outputs,
                #             sequence_length=stroke_lengths,
                #             dtype=tf.float32,
                #             scope="rnn_classification")
                #     layer_features.append(outputs)
                #     # outputs is [batch_size, L, N] where L is the maximal sequence length and N
                #     # the number of nodes in the last layer.
                #     mask = tf.tile(
                #         tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(outputs)[1]), 2),
                #         [1, 1, tf.shape(outputs)[2]])
                #     zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
                #     outputs = tf.reduce_sum(zero_outside, axis=1)
                #     layer_features.append(outputs)

            c_conv_encoder = outputs
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "RNNLayerNet")

        return c_conv_encoder, layer_features


class FCNConvSeqClassifier:
    def __init__(self, batch_size, layer_stage_sizes, rnn_cell_type, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, bidirectional=False, conv_rnn_sizes=[]):

        """
        Initializes a RNN Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.rnn_cell_type = rnn_cell_type
        self.bidirectional = bidirectional
        self.conv_rnn_sizes = conv_rnn_sizes


    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the network producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('FCNSeqLayerNet'):
                outputs = image_input

                for i in range(len(self.conv_rnn_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        if self.batch_norm_use:
                            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                        # Add dropout layer if enabled and not first convolution layer.
                        if i > 0:
                            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
                        outputs = tf.layers.conv1d(outputs,
                                                   filters=self.conv_rnn_sizes[i],
                                                   kernel_size=3,
                                                   activation=None,
                                                   strides=1,
                                                   padding='SAME',
                                                   name='conv1d_%d' % i)
                        layer_features.append(outputs)


                with tf.variable_scope('fcn_stage'):
                    for i in range(len(self.layer_stage_sizes)):
                        outputs = tf.layers.dense(outputs, units=self.layer_stage_sizes[i])
                        outputs = leaky_relu(outputs, name="leaky_relu{}".format(i))
                        layer_features.append(outputs)
                        if self.batch_norm_use:
                            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                center=True, is_training=training, renorm=False)
                        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            c_conv_encoder = outputs
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            self.reuse = True

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "FCNSeqLayerNet")

        return c_conv_encoder, layer_features


class CombinedClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes,
                 rnn_cell_type, rnn_stage_sizes=[128],bidirectional=False, conv_rnn_sizes=[],
                 num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, filter_size=[3, 3],
                 num_dense_layers=2, num_dense_units=1024):

        """
        Initializes a NN Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.filter_size = filter_size
        self.rnn_cell_type = rnn_cell_type
        self.bidirectional = bidirectional
        self.conv_rnn_sizes = conv_rnn_sizes
        self.rnn_stage_sizes = rnn_stage_sizes
        self.num_dense_layers = num_dense_layers
        self.num_dense_units = num_dense_units

    def __call__(self, image_input, seq_input, training=False, dropout_rate=0.0):
        """
        Runs the NN producing the predictions and the gradients.
        :param image_input: input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param seq_input: input to produce embeddings
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        used = tf.sign(tf.reduce_max(tf.abs(seq_input), 2))
        stroke_lengths = tf.cast(tf.reduce_sum(used, 1), tf.int32)
        # num_rnn_nodes = 128
        # num_rnn_layers = 3

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('CombNet'):
                image_outputs = image_input
                stroke_outputs = seq_input

                # CNN Part
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if (j == self.inner_layer_depth-1) and self.strided_dim_reduction:
                                    stride = 2
                                else:
                                    stride = 1
                                image_outputs = tf.layers.conv2d(image_outputs, self.layer_stage_sizes[i], self.filter_size,
                                                                 strides=(stride, stride),
                                                                 padding='SAME', activation=None)
                                image_outputs = leaky_relu(image_outputs, name="leaky_relu{}".format(i))
                                layer_features.append(image_outputs)
                                if self.batch_norm_use:
                                    image_outputs = batch_norm(image_outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                                layer_features.append(image_outputs)

                        if self.strided_dim_reduction==False:
                            image_outputs = tf.layers.max_pooling2d(image_outputs, pool_size=(2, 2), strides=2)

                        image_outputs = tf.layers.dropout(image_outputs, rate=dropout_rate, training=training)
                        # apply dropout only at dimensionality
                        # reducing steps, i.e. the last layer in
                        # every group


                # RNN part
                if self.bidirectional:
                    if self.rnn_cell_type == 'BasicRNNCell':
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell
                    elif self.rnn_cell_type == 'LSTMCell':
                        rnn_cell = tf.nn.rnn_cell.LSTMCell
                    elif self.rnn_cell_type == 'GRUCell':
                        rnn_cell = tf.nn.rnn_cell.GRUCell
                    else:
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell

                    for i in range(len(self.conv_rnn_sizes)):
                        with tf.variable_scope('conv_stage_{}'.format(i)):
                            if self.batch_norm_use:
                                stroke_outputs = batch_norm(stroke_outputs, decay=0.99, scale=True,
                                                            center=True, is_training=training, renorm=False)
                            # Add dropout layer if enabled and not first convolution layer.
                            if i > 0:
                                stroke_outputs = tf.layers.dropout(stroke_outputs, rate=dropout_rate, training=training)
                            stroke_outputs = tf.layers.conv1d(stroke_outputs,
                                                       filters=self.conv_rnn_sizes[i],
                                                       kernel_size=3,
                                                       activation=None,
                                                       strides=1,
                                                       padding='SAME',
                                                       name='conv1d_%d' % i)

                    with tf.variable_scope('rnn_stage'):
                        # apply dropout
                        cells_fw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.rnn_stage_sizes[i]), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.rnn_stage_sizes))]
                        cells_bw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.rnn_stage_sizes[i]), output_keep_prob=1.0 - dropout_rate) for i in range(len(self.rnn_stage_sizes))]
                        stroke_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=cells_fw,
                                cells_bw=cells_bw,
                                inputs=stroke_outputs,
                                sequence_length=stroke_lengths,
                                dtype=tf.float32,
                                scope="rnn_classification")

                        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
                        # the number of nodes in the last layer.
                        mask = tf.tile(
                            tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(stroke_outputs)[1]), 2),
                            [1, 1, tf.shape(stroke_outputs)[2]])
                        zero_outside = tf.where(mask, stroke_outputs, tf.zeros_like(stroke_outputs))
                        stroke_outputs = tf.reduce_sum(zero_outside, axis=1)
                else:
                    for i in range(len(self.rnn_stage_sizes)):
                        with tf.variable_scope('RNN_stage_{}'.format(i)):
                            if self.rnn_cell_type == 'BasicRNNCell':
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_stage_sizes[i])
                            elif self.rnn_cell_type == 'LSTMCell':
                                rnn_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_stage_sizes[i])
                            elif self.rnn_cell_type == 'GRUCell':
                                rnn_cell = tf.nn.rnn_cell.GRUCell(self.rnn_stage_sizes[i])
                            else:
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_stage_sizes[i])

                            # this is where dropout is applied in rnn tutorials

                            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=1.0 - dropout_rate)

                            stroke_outputs, state = tf.nn.dynamic_rnn(rnn_cell, stroke_outputs,
                                                               sequence_length=stroke_lengths,
                                                               dtype=tf.float32)

            image_outputs = tf.contrib.layers.flatten(image_outputs)
            stroke_outputs = tf.contrib.layers.flatten(stroke_outputs)
            c_encoder = tf.concat([image_outputs, stroke_outputs], axis=1)
            for i in range(self.num_dense_layers):
                c_encoder = tf.layers.dense(c_encoder, units=self.num_dense_units)
                c_encoder = leaky_relu(c_encoder, name="dense_leaky_relu{}".format(i))

                c_encoder = tf.layers.dropout(c_encoder, rate=dropout_rate, training=training)
            c_encoder = tf.layers.dense(c_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "СombNet")

        return c_encoder, layer_features


class MultitaskClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes,
                 rnn_cell_type, rnn_stage_sizes=[128],bidirectional=False, conv_rnn_sizes=[],
                 num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, filter_size=[3, 3],
                 num_dense_layers=2, num_dense_units=1024):

        """
        Initializes a NN Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.filter_size = filter_size
        self.rnn_cell_type = rnn_cell_type
        self.bidirectional = bidirectional
        self.conv_rnn_sizes = conv_rnn_sizes
        self.rnn_stage_sizes = rnn_stage_sizes
        self.num_dense_layers = num_dense_layers
        self.num_dense_units = num_dense_units

    def __call__(self, image_input, seq_input, training=False, dropout_rate=0.0, rnn_dropout=0.0):
        """
        Runs the NN producing the predictions and the gradients.
        :param image_input: input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param seq_input: input to produce embeddings
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        used = tf.sign(tf.reduce_max(tf.abs(seq_input), 2))
        stroke_lengths = tf.cast(tf.reduce_sum(used, 1), tf.int32)

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('CombNet'):
                image_outputs = image_input
                stroke_outputs = seq_input

                # CNN Part
                for i in range(len(self.layer_stage_sizes)):
                    with tf.variable_scope('conv_stage_{}'.format(i)):
                        for j in range(self.inner_layer_depth):
                            with tf.variable_scope('conv_{}_{}'.format(i, j)):
                                if (j == self.inner_layer_depth-1) and self.strided_dim_reduction:
                                    stride = 2
                                else:
                                    stride = 1
                                image_outputs = tf.layers.conv2d(image_outputs, self.layer_stage_sizes[i], self.filter_size,
                                                                 strides=(stride, stride),
                                                                 padding='SAME', activation=None)
                                image_outputs = leaky_relu(image_outputs, name="leaky_relu{}".format(i))
                                layer_features.append(image_outputs)
                                if self.batch_norm_use:
                                    image_outputs = batch_norm(image_outputs, decay=0.99, scale=True,
                                                         center=True, is_training=training, renorm=False)
                                layer_features.append(image_outputs)

                        if self.strided_dim_reduction==False:
                            image_outputs = tf.layers.max_pooling2d(image_outputs, pool_size=(2, 2), strides=2)

                        image_outputs = tf.layers.dropout(image_outputs, rate=dropout_rate, training=training)
                        # apply dropout only at dimensionality
                        # reducing steps, i.e. the last layer in
                        # every group


                # RNN part
                if self.bidirectional:
                    if self.rnn_cell_type == 'BasicRNNCell':
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell
                    elif self.rnn_cell_type == 'LSTMCell':
                        rnn_cell = tf.nn.rnn_cell.LSTMCell
                    elif self.rnn_cell_type == 'GRUCell':
                        rnn_cell = tf.nn.rnn_cell.GRUCell
                    else:
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell

                    for i in range(len(self.conv_rnn_sizes)):
                        with tf.variable_scope('conv_stage_{}'.format(i)):
                            if self.batch_norm_use:
                                stroke_outputs = batch_norm(stroke_outputs, decay=0.99, scale=True,
                                                            center=True, is_training=training, renorm=False)
                            # Add dropout layer if enabled and not first convolution layer.
                            if i > 0:
                                stroke_outputs = tf.layers.dropout(stroke_outputs, rate=rnn_dropout, training=training)
                            stroke_outputs = tf.layers.conv1d(stroke_outputs,
                                                       filters=self.conv_rnn_sizes[i],
                                                       kernel_size=3,
                                                       activation=None,
                                                       strides=1,
                                                       padding='SAME',
                                                       name='conv1d_%d' % i)

                    with tf.variable_scope('rnn_stage'):
                        # apply dropout
                        cells_fw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.rnn_stage_sizes[i]), output_keep_prob=1.0 - rnn_dropout) for i in range(len(self.rnn_stage_sizes))]
                        cells_bw = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell(self.rnn_stage_sizes[i]), output_keep_prob=1.0 - rnn_dropout) for i in range(len(self.rnn_stage_sizes))]
                        stroke_outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                cells_fw=cells_fw,
                                cells_bw=cells_bw,
                                inputs=stroke_outputs,
                                sequence_length=stroke_lengths,
                                dtype=tf.float32,
                                scope="rnn_classification")

                        # outputs is [batch_size, L, N] where L is the maximal sequence length and N
                        # the number of nodes in the last layer.
                        mask = tf.tile(
                            tf.expand_dims(tf.sequence_mask(stroke_lengths, tf.shape(stroke_outputs)[1]), 2),
                            [1, 1, tf.shape(stroke_outputs)[2]])
                        zero_outside = tf.where(mask, stroke_outputs, tf.zeros_like(stroke_outputs))
                        stroke_outputs = tf.reduce_sum(zero_outside, axis=1)
                else:
                    for i in range(len(self.rnn_stage_sizes)):
                        with tf.variable_scope('RNN_stage_{}'.format(i)):
                            if self.rnn_cell_type == 'BasicRNNCell':
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_stage_sizes[i])
                            elif self.rnn_cell_type == 'LSTMCell':
                                rnn_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_stage_sizes[i])
                            elif self.rnn_cell_type == 'GRUCell':
                                rnn_cell = tf.nn.rnn_cell.GRUCell(self.rnn_stage_sizes[i])
                            else:
                                rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_stage_sizes[i])

                            # this is where dropout is applied in rnn tutorials

                            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=1.0 - rnn_dropout)

                            stroke_outputs, state = tf.nn.dynamic_rnn(rnn_cell, stroke_outputs,
                                                               sequence_length=stroke_lengths,
                                                               dtype=tf.float32)

            image_outputs = tf.contrib.layers.flatten(image_outputs)
            stroke_outputs = tf.contrib.layers.flatten(stroke_outputs)
            comb_encoder = tf.concat([image_outputs, stroke_outputs], axis=1)
            for i in range(self.num_dense_layers):
                comb_encoder = tf.layers.dense(comb_encoder, units=self.num_dense_units)
                comb_encoder = leaky_relu(comb_encoder, name="dense_leaky_relu{}".format(i))
                comb_encoder = tf.layers.dropout(comb_encoder, rate=dropout_rate, training=training)

            image_encoder = tf.layers.dense(image_outputs, units=self.num_classes)
            stroke_encoder = tf.layers.dense(stroke_outputs, units=self.num_classes)
            comb_encoder = tf.layers.dense(comb_encoder, units=self.num_classes)


        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "СombNet")

        return image_encoder, stroke_encoder, comb_encoder, layer_features


class DenseNetClassifier:
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                 inner_layer_depth=1, strided_dim_reduction=True, filter_size=[3, 3]):

        """
        Initializes a VGG Classifier architecture
        :param batch_size: The size of the data batch
        :param layer_stage_sizes: A list containing the filters for each layer stage, where layer stage is a series of
        convolutional layers with stride=1 and no max pooling followed by a dimensionality reducing stage which is
        either a convolution with stride=1 followed by max pooling or a convolution with stride=2
        (i.e. strided convolution). So if we pass a list [64, 128, 256] it means that if we have inner_layer_depth=2
        then stage 0 will have 2 layers with stride=1 and filter size=64 and another dimensionality reducing convolution
        with either stride=1 and max pooling or stride=2 to dimensionality reduce. Similarly for the other stages.
        :param name: Name of the network
        :param num_classes: Number of classes we will need to classify
        :param num_channels: Number of channels of our image data.
        :param batch_norm_use: Whether to use batch norm between layers or not.
        :param inner_layer_depth: The amount of extra layers on top of the dimensionality reducing stage to have per
        layer stage.
        :param strided_dim_reduction: Whether to use strided convolutions instead of max pooling.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.filter_size = filter_size

        # these two hyperparameters may be changed
        self.nb_blocks = 2 # how many (dense block + Transition Layer) ?
        self.filters = 12


    def conv_layer(self, input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
            return network

    def global_Average_Pooling(self, x, stride=1):
        """
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
        It is global average pooling without tflearn
        """

        return global_avg_pool(x, name='Global_avg_pooling')
        # But maybe you need to install h5py and curses or not


    def batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True) :
            return tf.cond(training,
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))

    def drop_out(self, x, rate, training) :
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def relu(self, x):
        return tf.nn.relu(x)

    def average_pooling(self, x, pool_size=[2,2], stride=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def max_Pooling(self, x, pool_size=[3,3], stride=2, padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def concatenation(self, layers) :
        return tf.concat(layers, axis=3)

    def linear(self, x) :
        return tf.layers.dense(inputs=x, units=self.num_classes, name='linear')

    def bottleneck_layer(self, x, scope, dropout_rate, training):
        # print(x)
        with tf.name_scope(scope):
            x = self.batch_Normalization(x, training=training, scope=scope+'_batch1')
            x = self.relu(x)
            x = self.conv_layer(x, filter=4*self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = self.drop_out(x, rate=dropout_rate, training=training)

            x = self.batch_Normalization(x, training=training, scope=scope+'_batch2')
            x = self.relu(x)
            x = self.conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = self.drop_out(x, rate=dropout_rate, training=training)

            # print(x)

            return x

    def transition_layer(self, x, scope, dropout_rate, training):
        with tf.name_scope(scope):
            x = self.batch_Normalization(x, training=training, scope=scope+'_batch1')
            x = self.relu(x)
            x = self.conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = self.drop_out(x, rate=dropout_rate, training=training)
            x = self.average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name, dropout_rate, training):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0), dropout_rate=dropout_rate, training=training)

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = self.concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1), dropout_rate=dropout_rate, training=training)
                layers_concat.append(x)

            x = self.concatenation(layers_concat)

            return x

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """
        Runs the CNN producing the predictions and the gradients.
        :param image_input: Image input to produce embeddings for. e.g. for EMNIST [batch_size, 28, 28, 1]
        :param training: A flag indicating training or evaluation
        :param dropout_rate: A tf placeholder of type tf.float32 indicating the amount of dropout applied
        :return: Embeddings of size [batch_size, self.num_classes]
        """

        with tf.variable_scope(self.name, reuse=self.reuse):
            layer_features = []
            with tf.variable_scope('DenseNet'):
                outputs = image_input
                outputs = self.conv_layer(outputs, filter=2*self.filters, kernel=[7,7], stride=2, layer_name='conv0')
                outputs = self.max_Pooling(outputs, pool_size=[3,3], stride=2)

                for i in range(self.nb_blocks) :
                    # 6 -> 12 -> 48
                    outputs = self.dense_block(input_x=outputs, nb_layers=4, layer_name='dense_'+str(i), dropout_rate=dropout_rate, training=training)
                    outputs = self.transition_layer(outputs, scope='trans_'+str(i), dropout_rate=dropout_rate, training=training)

                """
                x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1', dropout_rate=dropout_rate, training=training)
                x = self.transition_layer(x, scope='trans_1', dropout_rate=dropout_rate, training=training)
                x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2', dropout_rate=dropout_rate, training=training)
                x = self.transition_layer(x, scope='trans_2', dropout_rate=dropout_rate, training=training)
                x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3', dropout_rate=dropout_rate, training=training)
                x = self.transition_layer(x, scope='trans_3', dropout_rate=dropout_rate, training=training)
                """

                outputs = self.dense_block(input_x=outputs, nb_layers=32, layer_name='dense_final', dropout_rate=dropout_rate, training=training)

                # 100 Layer
                outputs = self.batch_Normalization(outputs, training=training, scope='linear_batch')
                outputs = self.relu(outputs)
                outputs = self.global_Average_Pooling(outputs)
                outputs = flatten(outputs)
                outputs = self.linear(outputs)

                # x = tf.reshape(x, [-1, 10])

            c_conv_encoder = outputs
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "VGGNet")

        return c_conv_encoder, layer_features


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm_resnet(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                                 Should be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                                             data_format):
    """
    Convolution then batch normalization then ReLU as described by:
        Deep Residual Learning for Image Recognition
        https://arxiv.org/pdf/1512.03385.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm_resnet(inputs=shortcut, training=training,
                                                    data_format=data_format)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                                             data_format):
    """
    Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the convolutions.
        training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
        projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
        strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)

    return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                                                 strides, data_format):
    """
    Similar to _building_block_v1(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    """
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm_resnet(inputs=shortcut, training=training,
                                                    data_format=data_format)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                                                 strides, data_format):
    """
    Similar to _building_block_v2(), except using the "bottleneck" blocks
    described in:
        Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    adapted to the ordering conventions of:
        Batch normalization then ReLu then convolution as described by:
            Identity Mappings in Deep Residual Networks
            https://arxiv.org/pdf/1603.05027.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
    """
    shortcut = inputs
    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)

    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

    inputs = batch_norm_resnet(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                                training, name, data_format):
    """Creates one layer of blocks for the ResNet model.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
        filters: The number of filters for the first convolution of the layer.
        bottleneck: Is the block created a bottleneck block.
        block_fn: The block to use within the model, either `building_block` or
            `bottleneck_block`.
        blocks: The number of blocks contained in the layer.
        strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
        training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
        name: A string name for the tensor output of the block layer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block layer.
    """

    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                                        data_format)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return tf.identity(inputs, name)


class ResNetClassifier:
    """Base class for building the Resnet Model.
    """
    def __init__(self, batch_size, layer_stage_sizes, name, num_classes, num_channels=1, batch_norm_use=False,
                             inner_layer_depth=1, strided_dim_reduction=True, filter_size=[3, 3]):
    # def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
    #                            kernel_size,
    #                            conv_stride, first_pool_size, first_pool_stride,
    #                            second_pool_size, second_pool_stride, block_sizes, block_strides,
    #                            final_size, version=DEFAULT_VERSION, data_format=None):
        """Creates a model for classifying an image.
        Args:
            resnet_size: A single integer for the size of the ResNet model.
            bottleneck: Use regular blocks or bottleneck blocks.
            num_classes: The number of classes used as labels.
            num_filters: The number of filters to use for the first block layer
                of the model. This number is then doubled for each subsequent block
                layer.
            kernel_size: The kernel size to use for convolution.
            conv_stride: stride size for the initial convolutional layer
            first_pool_size: Pool size to be used for the first pooling layer.
                If none, the first pooling layer is skipped.
            first_pool_stride: stride size for the first pooling layer. Not used
                if first_pool_size is None.
            second_pool_size: Pool size to be used for the second pooling layer.
            second_pool_stride: stride size for the final pooling layer
            block_sizes: A list containing n values, where n is the number of sets of
                block layers desired. Each value should be the number of blocks in the
                i-th set.
            block_strides: List of integers representing the desired stride size for
                each of the sets of block layers. Should be same length as block_sizes.
            final_size: The expected size of the model after the second pooling.
            version: Integer representing which version of the ResNet network to use.
                See README for details. Valid values: [1, 2]
            data_format: Input format ('channels_last', 'channels_first', or None).
                If set to None, the format is dependent on whether a GPU is available.
        """
        # we will probably do only a couple of experiments with resnet,
        # so we just hardcode the values here - passing them will not work
        resnet_size = 110
        num_blocks = (resnet_size - 2) // 6
        bottleneck = False
        num_classes = 10
        num_filters = 16
        kernel_size = 3
        conv_stride = 1
        first_pool_size = None
        first_pool_stride = None
        second_pool_size = 7
        second_pool_stride = 1
        block_sizes = [num_blocks] * 3
        block_strides = [1, 2, 2]
        final_size = 64
        version = 2
        data_format = None
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_stage_sizes = layer_stage_sizes
        self.name = name
        self.num_classes = num_classes
        self.batch_norm_use = batch_norm_use
        self.inner_layer_depth = inner_layer_depth
        self.strided_dim_reduction = strided_dim_reduction
        self.build_completed = False
        self.filter_size = filter_size

        self.resnet_size = resnet_size

        if not data_format:
            data_format = (
                    'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.resnet_version = version
        if version not in (1, 2):
            raise ValueError(
                    "Resnet version should be 1 or 2. See README for citations.")

        self.bottleneck = bottleneck
        if bottleneck:
            if version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        else:
            if version == 1:
                self.block_fn = _building_block_v1
            else:
                self.block_fn = _building_block_v2

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.second_pool_size = second_pool_size
        self.second_pool_stride = second_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size

    def __call__(self, image_input, training=False, dropout_rate=0.0):
        """Add operations to classify a batch of input images.
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Set to True to add operations required only when
                training the classifier.
        Returns:
            A logits Tensor with shape [<batch_size>, self.num_classes].
        """
        with tf.variable_scope(self.name, reuse=self.reuse):
            inputs = image_input
            layer_features = []
            with tf.variable_scope('ResNet'):
                if self.data_format == 'channels_first':
                    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                    # This provides a large performance boost on GPU. See
                    # https://www.tensorflow.org/performance/performance_guide#data_formats
                    inputs = tf.transpose(inputs, [0, 3, 1, 2])

                inputs = conv2d_fixed_padding(
                        inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
                        strides=self.conv_stride, data_format=self.data_format)
                inputs = conv2d_fixed_padding(inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size, strides=self.conv_stride, data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_conv')

                if self.first_pool_size:
                    inputs = tf.layers.max_pooling2d(
                            inputs=inputs, pool_size=self.first_pool_size,
                            strides=self.first_pool_stride, padding='SAME',
                            data_format=self.data_format)
                    inputs = tf.identity(inputs, 'initial_max_pool')

                for i, num_blocks in enumerate(self.block_sizes):
                    num_filters = self.num_filters * (2**i)
                    inputs = block_layer(
                            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
                            block_fn=self.block_fn, blocks=num_blocks,
                            strides=self.block_strides[i], training=training,
                            name='block_layer{}'.format(i + 1), data_format=self.data_format)

                inputs = batch_norm_resnet(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)
                inputs = tf.layers.average_pooling2d(
                        inputs=inputs, pool_size=self.second_pool_size,
                        strides=self.second_pool_stride, padding='VALID',
                        data_format=self.data_format)
                inputs = tf.identity(inputs, 'final_avg_pool')

                inputs = tf.reshape(inputs, [-1, self.final_size])
                inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
                inputs = tf.identity(inputs, 'final_dense')

        # c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
        # c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.num_classes)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if not self.build_completed:
            self.build_completed = True
            count_parameters(self.variables, "ResNet")

        return inputs, layer_features
