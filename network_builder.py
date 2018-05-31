import tensorflow as tf
import math

from network_architectures import VGGClassifier, FCCLayerClassifier, RNNLayerClassifier, CombinedClassifier, MultitaskClassifier, FCNConvSeqClassifier, DenseNetClassifier, ResNetClassifier

class ClassifierNetworkGraph:
    def __init__(self, input_x, target_placeholder, dropout_rate,
                 batch_size=100, num_channels=1, n_classes=100, is_training=True, augment_rotate_flag=True,
                 tensorboard_use=False, use_batch_normalization=False, strided_dim_reduction=True,
                 network_name='VGG_classifier', layer_stage_sizes=[128], rnn_cell_type='BasicRNNCell',
                 inner_layer_depth=1, filter_size=[3, 3], input_seq=None, bidirectional=False, rnn_stage_sizes=[128],
                 conv_rnn_sizes=[], num_dense_layers=2, num_dense_units=1024):

        """
        Initializes a Classifier Network Graph that can build models, train, compute losses and save summary statistics
        and images
        :param input_x: A placeholder that will feed the input images, usually of size [batch_size, height, width,
        channels]
        :param target_placeholder: A target placeholder of size [batch_size,]. The classes should be in index form
               i.e. not one hot encoding, that will be done automatically by tf
        :param dropout_rate: A placeholder of size [None] that holds a single float that defines the amount of dropout
               to apply to the network. i.e. for 0.1 drop 0.1 of neurons
        :param batch_size: The batch size
        :param num_channels: Number of channels
        :param n_classes: Number of classes we will be classifying
        :param is_training: A placeholder that will indicate whether we are training or not
        :param augment_rotate_flag: A placeholder indicating whether to apply rotations augmentations to our input data
        :param tensorboard_use: Whether to use tensorboard in this experiment
        :param use_batch_normalization: Whether to use batch normalization between layers
        :param strided_dim_reduction: Whether to use strided dim reduction instead of max pooling
        :param input_seq: A placeholder that will optionally feed the input strokes
        """
        self.batch_size = batch_size
        if network_name == "VGG_classifier":
            self.c = VGGClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization,
                                   num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                   strided_dim_reduction=strided_dim_reduction,
                                   inner_layer_depth=inner_layer_depth,
                                   filter_size=filter_size)
            # layer_stage_sizes=[64, 128, 256], inner_layer_depth=2)
        elif network_name == "ResNetClassifier":
            self.c = ResNetClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization,
                                   num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                   strided_dim_reduction=strided_dim_reduction,
                                   inner_layer_depth=inner_layer_depth,
                                   filter_size=filter_size)
        elif network_name == "DenseNetClassifier":
            self.c = DenseNetClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization,
                                   num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                   strided_dim_reduction=strided_dim_reduction,
                                   inner_layer_depth=inner_layer_depth,
                                   filter_size=filter_size)
        elif network_name == "FCCClassifier":
            self.c = FCCLayerClassifier(self.batch_size, name="classifier_neural_network",
                                   batch_norm_use=use_batch_normalization,
                                   num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                   inner_layer_depth=inner_layer_depth)
            # layer_stage_sizes=[64, 128, 256]
        elif network_name == "RNNClassifier":
            self.c = RNNLayerClassifier(self.batch_size, name="classifier_neural_network",
                                        num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                        rnn_cell_type=rnn_cell_type,
                                        bidirectional=bidirectional,
                                        conv_rnn_sizes=conv_rnn_sizes)
        elif network_name == "FCNConvSeqClassifier":
            self.c = FCNConvSeqClassifier(self.batch_size, name="classifier_neural_network",
                                        num_classes=n_classes, layer_stage_sizes=layer_stage_sizes,
                                        rnn_cell_type=rnn_cell_type,
                                        bidirectional=bidirectional,
                                        conv_rnn_sizes=conv_rnn_sizes)
        elif network_name == "CombinedClassifier":
            self.c = CombinedClassifier(self.batch_size, name="classifier_neural_network",
                                        batch_norm_use=use_batch_normalization,
                                        num_classes=n_classes,
                                        layer_stage_sizes=layer_stage_sizes,
                                        strided_dim_reduction=strided_dim_reduction,
                                        inner_layer_depth=inner_layer_depth,
                                        filter_size=filter_size,
                                        rnn_cell_type=rnn_cell_type,
                                        rnn_stage_sizes=rnn_stage_sizes,
                                        bidirectional=bidirectional,
                                        conv_rnn_sizes=conv_rnn_sizes,
                                        num_dense_layers=num_dense_layers,
                                        num_dense_units=num_dense_units)

        self.input_x = input_x
        self.input_seq = input_seq
        self.dropout_rate = dropout_rate
        self.targets = target_placeholder

        self.training_phase = is_training
        self.n_classes = n_classes
        self.iterations_trained = 0

        self.augment_rotate = augment_rotate_flag
        self.is_tensorboard = tensorboard_use
        self.strided_dim_reduction = strided_dim_reduction
        self.use_batch_normalization = use_batch_normalization

        # this has been added here
        self.network_name = network_name



    def loss(self):
        """build models, calculates losses, saves summary statistcs and images.
        Returns:
            dict of losses.
        """
        with tf.name_scope("losses"):
            image_inputs = self.data_augment_batch(self.input_x)  # conditionally apply augmentaions
            if self.input_seq is not None:
                seq_inputs = self.input_seq # maybe we will apply augmentations here later on
            true_outputs = self.targets
            # produce predictions and get layer features to save for visual inspection
            # we need to pass here also the sequence input
            if self.input_seq is not None:
                preds, layer_features = self.c(image_input=image_inputs, seq_input=seq_inputs, training=self.training_phase,
                                               dropout_rate=self.dropout_rate)
            else:
                preds, layer_features = self.c(image_input=image_inputs, training=self.training_phase,
                                               dropout_rate=self.dropout_rate)
            # compute loss and accuracy
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(true_outputs, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            if self.network_name == "ResNetClassifier":
                # Add weight decay to the loss.
                weight_decay = 5e-4
                cross_entropy = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=preds))
                crossentropy_loss = cross_entropy + weight_decay * tf.add_n(
                    [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            else:
                crossentropy_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=preds))

            # add loss and accuracy to collections
            tf.add_to_collection('crossentropy_losses', crossentropy_loss)
            tf.add_to_collection('accuracy', accuracy)

            # save summaries for the losses, accuracy and image summaries for input images, augmented images
            # and the layer features
            if len(self.input_x.get_shape().as_list()) == 4:
                self.save_features(name="VGG_features", features=layer_features)
            tf.summary.image('image', [tf.concat(tf.unstack(self.input_x, axis=0), axis=0)])
            tf.summary.image('augmented_image', [tf.concat(tf.unstack(image_inputs, axis=0), axis=0)])
            tf.summary.scalar('crossentropy_losses', crossentropy_loss)
            tf.summary.scalar('accuracy', accuracy)

        return {"crossentropy_losses": tf.add_n(tf.get_collection('crossentropy_losses'),
                                                name='total_classification_loss'),
                "accuracy": tf.add_n(tf.get_collection('accuracy'), name='total_accuracy')}

    def save_features(self, name, features, num_rows_in_grid=4):
        """
        Saves layer features in a grid to be used in tensorboard
        :param name: Features name
        :param features: A list of feature tensors
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = num_rows_in_grid
            x_channels = int(channels / y_channels)

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                  y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)

    def rotate_image(self, image):
        """
        Rotates a single image
        :param image: An image to rotate
        :return: A rotated or a non rotated image depending on the result of the flip
        """
        # with 0.5 prob flip/reflect the image from left to right
        image = tf.image.random_flip_left_right(image, seed=None)

        # rotate the image by a random small angle in radians - 30 deg is about 0.5
        # update: rotate only by max. 0.2 radians, which is about 11 deg
        angle = tf.unstack(
            tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32, seed=None,
                              name=None))  # get a random number between -0.2 and 0.2
        image = tf.contrib.image.rotate(image, angle, interpolation='BILINEAR')

        # # now do the other things that were supposed to be happening there.
        # no_rotation_flip = tf.unstack(
        #     tf.random_uniform([1], minval=1, maxval=100, dtype=tf.int32, seed=None,
        #                       name=None))  # get a random number between 1 and 100
        # flip_boolean = tf.less_equal(no_rotation_flip[0], 0) # do not do rotations now
        # # if that number is less than or equal to 50 then set to true
        # random_variable = tf.unstack(tf.random_uniform([1], minval=1, maxval=3, dtype=tf.int32, seed=None, name=None))
        # # get a random variable between 1 and 3 for how many degrees the rotation will be i.e. k=1 means 1*90,
        # # k=2 2*90 etc.
        # image = tf.cond(flip_boolean, lambda: tf.image.rot90(image, k=random_variable[0]),
        #                 lambda: image)  # if flip_boolean is true the rotate if not then do not rotate
        return image

    def rotate_batch(self, batch_images):
        """
        Rotate a batch of images
        :param batch_images: A batch of images
        :return: A rotated batch of images (some images will not be rotated if their rotation flip ends up False)
        """
        shapes = map(int, list(batch_images.get_shape()))
        if len(list(batch_images.get_shape())) < 4:
            return batch_images
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked = tf.unstack(batch_images)
            new_images = []
            for image in batch_images_unpacked:
                new_images.append(self.rotate_image(image))
            new_images = tf.stack(new_images)
            new_images = tf.reshape(new_images, (batch_size, x, y, c))
            return new_images

    def data_augment_batch(self, batch_images):
        """
        Augments data with a variety of augmentations, in the current state only does rotations.
        :param batch_images: A batch of images to augment
        :return: Augmented data
        """
        batch_images = tf.cond(self.augment_rotate, lambda: self.rotate_batch(batch_images), lambda: batch_images)
        return batch_images

    def train(self, losses, learning_rate=1e-3, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        if self.network_name == "ResNetClassifier":
            c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
            # c_opt = tf.train.MomentumOptimizer(learning_rate=0.001,
            #                                    momentum=0.9)
        else:
            c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            c_error_opt_op = c_opt.minimize(losses["crossentropy_losses"], var_list=self.c.variables,
                                            colocate_gradients_with_ops=True)

        return c_error_opt_op

    def init_train(self):
        """
        Builds graph ops and returns them
        :return: Summary, losses and training ops
        """
        losses_ops = self.loss()
        c_error_opt_op = self.train(losses_ops)
        summary_op = tf.summary.merge_all()
        return summary_op, losses_ops, c_error_opt_op


class MultutaskNetworkGraph(ClassifierNetworkGraph):
    def __init__(self, input_x, input_seq, target_placeholder, dropout_rate, rnn_dropout,
                 batch_size=100, num_channels=1, n_classes=100, is_training=True, augment_rotate_flag=True,
                 tensorboard_use=False, use_batch_normalization=False, strided_dim_reduction=True,
                 network_name='VGG_classifier', layer_stage_sizes=[128], rnn_cell_type='BasicRNNCell',
                 inner_layer_depth=1, filter_size=[3, 3], bidirectional=False, rnn_stage_sizes=[128],
                 conv_rnn_sizes=[], num_dense_layers=2, num_dense_units=1024):

        """
        Initializes a Classifier Network Graph that can build models, train, compute losses and save summary statistics
        and images
        :param input_x: A placeholder that will feed the input images, usually of size [batch_size, height, width,
        channels]
        :param target_placeholder: A target placeholder of size [batch_size,]. The classes should be in index form
               i.e. not one hot encoding, that will be done automatically by tf
        :param dropout_rate: A placeholder of size [None] that holds a single float that defines the amount of dropout
               to apply to the network. i.e. for 0.1 drop 0.1 of neurons
        :param batch_size: The batch size
        :param num_channels: Number of channels
        :param n_classes: Number of classes we will be classifying
        :param is_training: A placeholder that will indicate whether we are training or not
        :param augment_rotate_flag: A placeholder indicating whether to apply rotations augmentations to our input data
        :param tensorboard_use: Whether to use tensorboard in this experiment
        :param use_batch_normalization: Whether to use batch normalization between layers
        :param strided_dim_reduction: Whether to use strided dim reduction instead of max pooling
        :param input_seq: A placeholder that will optionally feed the input strokes
        """
        self.batch_size = batch_size

        self.c = MultitaskClassifier(self.batch_size, name="classifier_neural_network",
                                    batch_norm_use=use_batch_normalization,
                                    num_classes=n_classes,
                                    layer_stage_sizes=layer_stage_sizes,
                                    strided_dim_reduction=strided_dim_reduction,
                                    inner_layer_depth=inner_layer_depth,
                                    filter_size=filter_size,
                                    rnn_cell_type=rnn_cell_type,
                                    rnn_stage_sizes=rnn_stage_sizes,
                                    bidirectional=bidirectional,
                                    conv_rnn_sizes=conv_rnn_sizes,
                                    num_dense_layers=num_dense_layers,
                                    num_dense_units=num_dense_units)

        self.input_x = input_x
        self.input_seq = input_seq
        self.dropout_rate = dropout_rate
        self.rnn_dropout = rnn_dropout
        self.targets = target_placeholder

        self.training_phase = is_training
        self.n_classes = n_classes
        self.iterations_trained = 0

        self.augment_rotate = augment_rotate_flag
        self.is_tensorboard = tensorboard_use
        self.strided_dim_reduction = strided_dim_reduction
        self.use_batch_normalization = use_batch_normalization


    def loss(self):
        """build models, calculates losses, saves summary statistcs and images.
        Returns:
            dict of losses.
        """
        with tf.name_scope("losses"):
            image_inputs = self.data_augment_batch(self.input_x)  # conditionally apply augmentaions
            seq_inputs = self.input_seq # maybe we will apply augmentations here later on
            true_outputs = self.targets

            # produce predictions and get layer features to save for visual inspection
            # we need to pass here also the sequence input
            image_preds, stroke_preds, comb_preds, layer_features = self.c(image_input=image_inputs,
                                                seq_input=seq_inputs, training=self.training_phase,
                                                dropout_rate=self.dropout_rate, rnn_dropout=self.rnn_dropout)

            # compute loss and accuracy
            correct_image_prediction = tf.equal(tf.argmax(image_preds, 1), tf.cast(true_outputs, tf.int64))
            correct_stroke_prediction = tf.equal(tf.argmax(stroke_preds, 1), tf.cast(true_outputs, tf.int64))
            correct_comb_prediction = tf.equal(tf.argmax(comb_preds, 1), tf.cast(true_outputs, tf.int64))

            accuracy_image = tf.reduce_mean(tf.cast(correct_image_prediction, tf.float32))
            accuracy_stroke = tf.reduce_mean(tf.cast(correct_stroke_prediction, tf.float32))
            accuracy_comb = tf.reduce_mean(tf.cast(correct_comb_prediction, tf.float32))

            crossentropy_image_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=image_preds))
            crossentropy_stroke_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=stroke_preds))
            crossentropy_comb_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_outputs, logits=comb_preds))

            # add loss and accuracy to collections
            tf.add_to_collection('crossentropy_image_losses', crossentropy_image_loss)
            tf.add_to_collection('accuracy_image', accuracy_image)
            tf.add_to_collection('crossentropy_stroke_losses', crossentropy_stroke_loss)
            tf.add_to_collection('accuracy_stroke', accuracy_stroke)
            tf.add_to_collection('crossentropy_comb_losses', crossentropy_comb_loss)
            tf.add_to_collection('accuracy_comb', accuracy_comb)

            # save summaries for the losses, accuracy and image summaries for input images, augmented images
            # and the layer features
            if len(self.input_x.get_shape().as_list()) == 4:
                self.save_features(name="Multitask_features", features=layer_features)
            tf.summary.image('image', [tf.concat(tf.unstack(self.input_x, axis=0), axis=0)])
            tf.summary.image('augmented_image', [tf.concat(tf.unstack(image_inputs, axis=0), axis=0)])

            tf.summary.scalar('crossentropy_image_losses', crossentropy_image_loss)
            tf.summary.scalar('accuracy_image', accuracy_image)
            tf.summary.scalar('crossentropy_stroke_losses', crossentropy_stroke_loss)
            tf.summary.scalar('accuracy_stroke', accuracy_stroke)
            tf.summary.scalar('crossentropy_comb_losses', crossentropy_comb_loss)
            tf.summary.scalar('accuracy_comb', accuracy_comb)

        return {"crossentropy_image_losses": tf.add_n(tf.get_collection('crossentropy_image_losses'),
                                                name='total_classification_image_loss'),
                "accuracy_image": tf.add_n(tf.get_collection('accuracy_image'), name='total_image_accuracy'),
                "crossentropy_stroke_losses": tf.add_n(tf.get_collection('crossentropy_stroke_losses'),
                                                        name='total_classification_stroke_loss'),
                "accuracy_stroke": tf.add_n(tf.get_collection('accuracy_stroke'), name='total_stroke_accuracy'),
                "crossentropy_comb_losses": tf.add_n(tf.get_collection('crossentropy_comb_losses'),
                                                        name='total_classification_comb_loss'),
                "accuracy_comb": tf.add_n(tf.get_collection('accuracy_comb'), name='total_comb_accuracy')}


    def train(self, losses, learning_rate=1e-3, beta1=0.9):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        c_opt = tf.train.AdamOptimizer(beta1=beta1, learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            c_error_opt_op = {"image_error_op": c_opt.minimize(losses["crossentropy_image_losses"], var_list=self.c.variables,
                                               colocate_gradients_with_ops=True),
                              "stroke_error_op": c_opt.minimize(losses["crossentropy_stroke_losses"], var_list=self.c.variables,
                                               colocate_gradients_with_ops=True),
                              "comb_error_op": c_opt.minimize(losses["crossentropy_comb_losses"], var_list=self.c.variables,
                                               colocate_gradients_with_ops=True)}

        return c_error_opt_op

    def init_train(self):
        """
        Builds graph ops and returns them
        :return: Summary, losses and training ops
        """
        losses_ops = self.loss()
        c_error_opt_op = self.train(losses_ops)
        summary_op = tf.summary.merge_all()
        return summary_op, losses_ops, c_error_opt_op
