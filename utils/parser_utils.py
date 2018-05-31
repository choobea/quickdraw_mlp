class ParserClass(object):
    def __init__(self, parser):
        """
        Parses arguments and saves them in the Parser Class
        :param parser: A parser to get input from
        """
        parser.add_argument('--batch_size', nargs="?", type=int, default=64, help='batch_size for experiment')
        parser.add_argument('--epochs', type=int, nargs="?", default=100, help='Number of epochs to train for')
        parser.add_argument('--logs_path', type=str, nargs="?", default="classification_logs/",
                            help='Experiment log path, '
                                 'where tensorboard is saved, '
                                 'along with .csv of results')
        parser.add_argument('--experiment_prefix', nargs="?", type=str, default="classification",
                            help='Experiment name without hp details')
        parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help="ID of epoch to continue from, "
                                                                                      "-1 means from scratch")
        parser.add_argument('--tensorboard_use', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--dropout_rate', nargs="?", type=float, default=0.0, help="Dropout value")
        parser.add_argument('--rnn_dropout_rate', nargs="?", type=float, default=0.0, help="RNN dropout value")
        parser.add_argument('--batch_norm_use', nargs="?", type=str, default="False", help='Whether to use tensorboard')
        parser.add_argument('--strided_dim_reduction', nargs="?", type=str, default="False",
                            help='Whether to use tensorboard')
        parser.add_argument('--seed', nargs="?", type=int, default=1122017, help='Whether to use tensorboard')

        parser.add_argument('--layer_stage_sizes', nargs='*', type=int, default=[128], help='Specify the sizes of layer stages')

        parser.add_argument('--num_classes_use', nargs='?', type=int, default=10, help='How many classes to use')

        # cnn arguments
        parser.add_argument('--inner_layer_depth', nargs='?', type=int, default=1, help='Specify the inner layer depth')
        parser.add_argument('--filter_size', nargs='*', type=int, default=[3,3], help='Specify the filter sizes')

        parser.add_argument('--network_name', nargs='?', type=str, default="conv", help='Specify the network name')

        # rnn arguments
        parser.add_argument('--rnn_cell_type', nargs='?', type=str, default="BasicRNNCell", help='Specify the type of RNN cell')
        parser.add_argument('--bidirectional', nargs='?', type=str, default="False", help='Specify if to use bidirectional layers')
        parser.add_argument('--conv_rnn_sizes', nargs='*', type=int, default=[], help='Specify the sizes of filters of convolutions for rnn in stages')
        parser.add_argument('--rnn_stage_sizes', nargs='*', type=int, default=[128], help='Specify the sizes of RNN stages')

        parser.add_argument('--num_dense_layers', nargs='?', type=int, default=2, help='Specify the number of dense layers')
        parser.add_argument('--num_dense_units', nargs='?', type=int, default=1024, help='Specify the number of dense units')

        # rotate data
        parser.add_argument('--rotate', nargs='?', type=str, default="False", help='Specify if to augment the data')

        self.args = parser.parse_args()

    def get_argument_variables(self):
        """
        Processes the parsed arguments and produces variables of specific types needed for the experiments
        :return: Arguments needed for experiments
        """
        batch_size = self.args.batch_size
        experiment_prefix = self.args.experiment_prefix
        strided_dim_reduction = True if self.args.strided_dim_reduction == "True" else False
        batch_norm = True if self.args.batch_norm_use == "True" else False
        seed = self.args.seed
        dropout_rate = self.args.dropout_rate
        rnn_dropout_rate = self.args.rnn_dropout_rate
        tensorboard_enable = True if self.args.tensorboard_use == "True" else False
        continue_from_epoch = self.args.continue_from_epoch  # use -1 to start from scratch
        epochs = self.args.epochs
        logs_path = self.args.logs_path

        layer_stage_sizes = self.args.layer_stage_sizes
        rnn_cell_type = self.args.rnn_cell_type
        bidirectional = True if self.args.bidirectional == "True" else False
        rnn_stage_sizes = self.args.rnn_stage_sizes
        conv_rnn_sizes = self.args.conv_rnn_sizes
        num_classes_use = self.args.num_classes_use
        inner_layer_depth = self.args.inner_layer_depth
        filter_size = self.args.filter_size
        num_dense_layers = self.args.num_dense_layers
        num_dense_units = self.args.num_dense_units
        network_name = self.args.network_name
        rotate = True if self.args.rotate == "True" else False

        return batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable, batch_norm, \
               strided_dim_reduction, experiment_prefix, dropout_rate, rnn_dropout_rate, layer_stage_sizes, rnn_cell_type, \
               bidirectional, rnn_stage_sizes, conv_rnn_sizes, num_classes_use, inner_layer_depth, filter_size, \
               num_dense_layers, num_dense_units, network_name, rotate
