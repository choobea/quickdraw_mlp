import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

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
