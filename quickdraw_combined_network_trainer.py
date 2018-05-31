import argparse
import numpy as np
import tensorflow as tf
import tqdm
from data_providers import QuickDrawCombinedDataProvider
from network_builder import ClassifierNetworkGraph
from utils.parser_utils import ParserClass
from utils.storage import build_experiment_folder, save_statistics

tf.reset_default_graph()  # resets any previous graphs to clear memory
parser = argparse.ArgumentParser(description='Welcome to CNN experiments script')  # generates an argument parser
parser_extractor = ParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable, batch_norm, \
strided_dim_reduction, experiment_prefix, dropout_rate_value, rnn_dropout_rate_value, layer_stage_sizes, \
rnn_cell_type, bidirectional, rnn_stage_sizes, conv_rnn_sizes, num_classes_use, inner_layer_depth, \
filter_size, num_dense_layers, num_dense_units, network_name, rotate = parser_extractor.get_argument_variables()

# returns a list of objects that contain
# our parsed input

#convnet_desc = ""
# if batch_norm:
#     convnet_desc = convnet_desc + "BN"
#
# for ls in layer_stage_sizes:
#     convnet_desc = "{}_{}".format(convnet_desc, ls)
#
# if network_name == 'fcn':
#     experiment_name = "exp{}_{}_{}_layers".format(experiment_prefix, network_name,
#                                                  len(layer_stage_sizes))
# elif network_name == 'convn':
#     experiment_name = "exp{}_{}_{}_layers_{}_filter".format(experiment_prefix, network_name,
#                                                             convnet_desc, max(filter_size))
rng = np.random.RandomState(seed=seed)  # set seed

convnet_desc = ""
if batch_norm:
    convnet_desc = convnet_desc + "BN"

for ls in layer_stage_sizes:
    convnet_desc = "{}_{}".format(convnet_desc, ls)

if bidirectional:
    if len(conv_rnn_sizes) == 0:
        experiment_name = "exp{}_{}_{}_layers_bidirectional_{}_{}_dense({})".format(experiment_prefix, rnn_cell_type,
                                                     len(rnn_stage_sizes), dropout_rate_value, num_dense_layers, num_dense_units)
    else:
        experiment_name = "exp{}_{}_{}_layers_{}_rnnconv_bidirectional_{}_dropout_{}_dense({})".format(experiment_prefix, rnn_cell_type,
                                                     len(rnn_stage_sizes), len(conv_rnn_sizes), dropout_rate_value, num_dense_layers,num_dense_units)

if bidirectional:
    if len(conv_rnn_sizes) == 0:
        experiment_name = "exp{}_{}_{}_rnnlayers_bidirectional_{}_dropout_{}_{}_convlayers_{}_filter_{}_dense({})".format(experiment_prefix, rnn_cell_type,
            len(rnn_stage_sizes), dropout_rate_value, network_name, convnet_desc, max(filter_size), num_dense_layers, num_dense_units)
    else:
        experiment_name = "exp{}_{}_{}_rnnlayers_{}_rnnconv_bidirectional_{}_dropout_{}_{}_convlayers_{}_filter_{}_dense({})".format(experiment_prefix, rnn_cell_type,
            len(rnn_stage_sizes), len(conv_rnn_sizes), dropout_rate_value, network_name, convnet_desc, max(filter_size), num_dense_layers, num_dense_units)
else:
    experiment_name = "exp{}_{}_{}_rnnlayers_{}_dropout_{}_{}_convlayers_{}_filter_{}_dense({})".format(experiment_prefix, rnn_cell_type,
                                                 len(rnn_stage_sizes), dropout_rate_value, network_name, convnet_desc, max(filter_size), num_dense_layers, num_dense_units)

network_name = "CombinedClassifier"

train_data = QuickDrawCombinedDataProvider(which_set="train", batch_size=batch_size, rng=rng, num_classes_use=num_classes_use)
val_data = QuickDrawCombinedDataProvider(which_set="valid", batch_size=batch_size, rng=rng, num_classes_use=num_classes_use)
test_data = QuickDrawCombinedDataProvider(which_set="test", batch_size=batch_size, rng=rng, num_classes_use=num_classes_use)
#  setup our data providers

print("Running {}".format(experiment_name))
print("Starting from epoch {}".format(continue_from_epoch))

saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir

# Placeholder setup
data_inputs_seq = tf.placeholder(tf.float32, [batch_size, train_data.inputs[0][0].shape[0], train_data.inputs[0][0].shape[1]], 'data-inputs-seq')
data_inputs_im = tf.placeholder(tf.float32, [batch_size, train_data.inputs[0][1].shape[0], train_data.inputs[0][1].shape[1], train_data.inputs[0][1].shape[2]], 'data-inputs-im')
data_targets = tf.placeholder(tf.int32, [batch_size], 'data-targets')

training_phase = tf.placeholder(tf.bool, name='training-flag')
rotate_data = tf.placeholder(tf.bool, name='rotate-flag')
dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')


classifier_network = ClassifierNetworkGraph(input_x=data_inputs_im, target_placeholder=data_targets,
                                            dropout_rate=dropout_rate, batch_size=batch_size,
                                            num_channels=1, n_classes=train_data.num_classes,
                                            is_training=training_phase, augment_rotate_flag=rotate_data,
                                            strided_dim_reduction=strided_dim_reduction,
                                            use_batch_normalization=batch_norm,
                                            network_name=network_name, layer_stage_sizes=layer_stage_sizes,
                                            inner_layer_depth=inner_layer_depth, filter_size=filter_size,
                                            rnn_cell_type=rnn_cell_type,
                                            input_seq=data_inputs_seq,
                                            bidirectional=bidirectional,
                                            rnn_stage_sizes=rnn_stage_sizes,
                                            conv_rnn_sizes=conv_rnn_sizes,
                                            num_dense_layers=num_dense_layers
                                            )  # initialize our computational graph
                                            # FCCClassifier VGG_classifier

if continue_from_epoch == -1:  # if this is a new experiment and not continuation of a previous one then generate a new
    # statistics file
    save_statistics(logs_filepath, "result_summary_statistics", ["epoch", "train_c_loss", "train_c_accuracy",
                                                                 "val_c_loss", "val_c_accuracy",
                                                                 "test_c_loss", "test_c_accuracy"], create=True)

start_epoch = continue_from_epoch if continue_from_epoch != -1 else 0  # if new experiment start from 0 otherwise
# continue where left off

summary_op, losses_ops, c_error_opt_op = classifier_network.init_train()  # get graph operations (ops)

total_train_batches = train_data.num_batches
total_val_batches = val_data.num_batches
total_test_batches = test_data.num_batches

best_epoch = 0

if tensorboard_enable:
    print("saved tensorboard file at", logs_filepath)
    writer = tf.summary.FileWriter(logs_filepath, graph=tf.get_default_graph())

init = tf.global_variables_initializer()  # initialization op for the graph

def split_x_batch(x_batch):
    """This is a helper function that splits the x_batch into batch of sequences
    and images.
    : param x_batch: input batch of shape [batch_size, ]. In each sub element
    we have a list of two items - sequence and image representation.

    This function takes the representations and splits them.
    """
    seq_inputs = np.empty((0, 70, 3))
    im_inputs = np.empty((0, 28, 28, 1))

    for ii in range(x_batch.shape[0]):
        seq_inputs = np.append(seq_inputs, [x_batch[ii][0]], axis=0)
        im_inputs = np.append(im_inputs, [x_batch[ii][1]], axis=0)

    return seq_inputs, im_inputs


with tf.Session() as sess:
    sess.run(init)  # actually running the initialization op
    train_saver = tf.train.Saver()  # saver object that will save our graph so we can reload it later for continuation of
    val_saver = tf.train.Saver()
    #  training or inference

    if continue_from_epoch != -1:
        train_saver.restore(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name,
                                                   continue_from_epoch-1))  # restore previous graph to continue operations

    best_val_accuracy = 0.
    with tqdm.tqdm(total=epochs - start_epoch) as epoch_pbar:
        for e in range(start_epoch, epochs):
            total_c_loss = 0.
            total_accuracy = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar_train:
                for batch_idx, (x_batch, y_batch) in enumerate(train_data):
                    # split x_batch into x_batch_im and x_batch_seq
                    x_batch_seq, x_batch_im = split_x_batch(x_batch)
                    iter_id = e * total_train_batches + batch_idx
                    _, c_loss_value, acc = sess.run(
                        [c_error_opt_op, losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
                        feed_dict={dropout_rate: dropout_rate_value, data_inputs_seq: x_batch_seq, data_inputs_im: x_batch_im,
                                   data_targets: y_batch, training_phase: True, rotate_data: rotate})
                    # Here we execute the c_error_opt_op which trains the network and also the ops that compute the
                    # loss and accuracy, we save those in _, c_loss_value and acc respectively.
                    total_c_loss += c_loss_value  # add loss of current iter to sum
                    total_accuracy += acc # add acc of current iter to sum

                    iter_out = "iter_num: {}, train_loss: {}, train_accuracy: {}".format(iter_id,
                                                                                         total_c_loss / (batch_idx + 1),
                                                                                         total_accuracy / (
                                                                                             batch_idx + 1)) # show
                                            # iter statistics using running averages of previous iter within this epoch
                    pbar_train.set_description(iter_out)
                    pbar_train.update(1)
                    if tensorboard_enable and batch_idx % 25 == 0:  # save tensorboard summary every 25 iterations
                        _summary = sess.run(
                            summary_op,
                            feed_dict={dropout_rate: dropout_rate_value, data_inputs_seq: x_batch_seq, data_inputs_im: x_batch_im,
                                       data_targets: y_batch, training_phase: True, rotate_data: rotate})
                        writer.add_summary(_summary, global_step=iter_id)

            total_c_loss /= total_train_batches  # compute mean of los
            total_accuracy /= total_train_batches # compute mean of accuracy

            save_path = train_saver.save(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
            # save graph and weights
            print("Saved current model at", save_path)

            total_val_c_loss = 0.
            total_val_accuracy = 0. #  run validation stage, note how training_phase placeholder is set to False
            # and that we do not run the c_error_opt_op which runs gradient descent, but instead only call the loss ops
            #  to collect losses on the validation set
            with tqdm.tqdm(total=total_val_batches) as pbar_val:
                for batch_idx, (x_batch, y_batch) in enumerate(val_data):
                    # split x_batch into x_batch_im and x_batch_seq
                    x_batch_seq, x_batch_im = split_x_batch(x_batch)
                    c_loss_value, acc = sess.run(
                        [losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
                        feed_dict={dropout_rate: dropout_rate_value, data_inputs_seq: x_batch_seq, data_inputs_im: x_batch_im,
                                   data_targets: y_batch, training_phase: False, rotate_data: False})
                    total_val_c_loss += c_loss_value
                    total_val_accuracy += acc
                    iter_out = "val_loss: {}, val_accuracy: {}".format(total_val_c_loss / (batch_idx + 1),
                                                                       total_val_accuracy / (batch_idx + 1))
                    pbar_val.set_description(iter_out)
                    pbar_val.update(1)

            total_val_c_loss /= total_val_batches
            total_val_accuracy /= total_val_batches

            if best_val_accuracy < total_val_accuracy:  # check if val acc better than the previous best and if
                # so save current as best and save the model as the best validation model to be used on the test set
                #  after the final epoch
                best_val_accuracy = total_val_accuracy
                best_epoch = e
                save_path = val_saver.save(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
                print("Saved best validation score model at", save_path)

            epoch_pbar.update(1)
            # save statistics of this epoch, train and val without test set performance
            save_statistics(logs_filepath, "result_summary_statistics",
                            [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy,
                             -1, -1])

        val_saver.restore(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, best_epoch))
        # restore model with best performance on validation set
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        # computer test loss and accuracy and save
        with tqdm.tqdm(total=total_test_batches) as pbar_test:
            for batch_id, (x_batch, y_batch) in enumerate(test_data):
                # split x_batch into x_batch_im and x_batch_seq
                x_batch_seq, x_batch_im = split_x_batch(x_batch)
                c_loss_value, acc = sess.run(
                    [losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
                    feed_dict={dropout_rate: dropout_rate_value, data_inputs_seq: x_batch_seq, data_inputs_im: x_batch_im,
                               data_targets: y_batch, training_phase: False, rotate_data: False})
                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
                iter_out = "test_loss: {}, test_accuracy: {}".format(total_test_c_loss / (batch_idx + 1),
                                                                     acc / (batch_idx + 1))
                pbar_test.set_description(iter_out)
                pbar_test.update(1)

        total_test_c_loss /= total_test_batches
        total_test_accuracy /= total_test_batches

        save_statistics(logs_filepath, "result_summary_statistics",
                        ["test set performance", -1, -1, -1, -1,
                         total_test_c_loss, total_test_accuracy])
