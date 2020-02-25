# @Author:LiuZhQ

import os
import sys
import logging
import shutil
from pyhocon import ConfigFactory


class BaseSetting(object):
    """
        This class include the setting of arguments and the tools of experimental recorder.
    """

    def __init__(self, config_path):
        super(BaseSetting, self).__init__()
        # Load config file
        self.con = ConfigFactory.parse_file(config_path)
        # Dataset
        self.data_name = self.con.dataset['name']  # Dataset name
        self.data_path = self.con.dataset['root']  # The root path of dataset
        self.num_classes = self.con.dataset['num_classes']  # The number of classes
        self.batch_size = self.con.dataset['batch_size']  # Batch size
        self.num_workers = self.con.dataset['num_workers']  # The number of workers to load dataset
        # Training mechanism
        self.repeat = self.con.mechanism['repeat']  # Repeat times of Experiment
        self.log_interval = self.con.mechanism['log_interval']  # Log interval
        self.epochs = self.con.mechanism['epochs']  # Training epochs
        self.output_path = self.con.mechanism['output_path']  # The output path of experimental logs
        self.experiment_id = self.con.mechanism['experiment_id']  # Experimental id
        self.seed = self.con.mechanism['seed']  # Random seed
        # Training strategy
        self.init_channels = self.con.strategy['init_channels']  # The number of initial channels
        self.layers = self.con.strategy['layers']  # The number of total layer
        self.cutout = self.con.strategy['cutout']  # Use cutout or not
        self.cutout_length = self.con.strategy['cutout_length']  # Cutout length
        self.grad_clip = self.con.strategy['grad_clip']  # Gradient clipping
        # Optimizer
        self.lr = self.con.optimizer.model['learning_rate']  # Learning rate
        self.lr_min = self.con.optimizer.model['learning_rate_min']  # Minimum learning rate
        self.momentum = self.con.optimizer.model['momentum']  # Momentum
        self.weight_decay = self.con.optimizer.model['weight_decay']  # Weight decay

    def get_logger(self, logger_name='search'):
        """
        Get experimental logger.
        :param logger_name: For multiple repetitive experiments, we need different loggers.
        :return: logger
        """
        logger = logging.getLogger(logger_name)
        log_format = '%(asctime)s ==> %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')

        fh = logging.FileHandler(os.path.join(self.output_path, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logger.addHandler(fh)
        return logger

    def set_output_path(self, flag=None):
        """
        Set output path and create directory if necessary.
        :param flag: Extra experimental identity
        :return:
        """
        if flag is not None:
            self.experiment_id = '{}_{}'.format(self.experiment_id, flag)
        self.output_path = os.path.join(self.output_path, self.experiment_id)
        # If output directories exist, judge delete them or not
        if os.path.exists(self.output_path):
            print("==> File exist: {}".format(self.output_path))
            action = input("Select Action: d (delete) / q (quit):").lower().strip()
            act = action
            if act == 'd':
                shutil.rmtree(self.output_path)
            else:
                raise OSError("==> File exist: {}".format(self.output_path))
        # If output directories does not exist, create them
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def save_setting(self):
        """
        Write arguments to a setting file.
        :return:
        """
        with open(os.path.join(self.output_path, "settings.log"), "w") as f:
            for k, v in vars(self).items():
                line = str(k).center(20, '=') + '=>' + str(v) + "\n"
                print(line, end='')
                f.write(line)

    def save_code(self, config_path, src=os.path.abspath("../"), dst=None):
        """
        Save experimental codes.
        :param config_path: Config path of this experiment
        :param src: The root path of codes to be saved
        :param dst: The destination of codes to be saved
        :return:
        """
        if dst is None:
            dst = os.path.join(self.output_path, 'code')
        for f in os.listdir(src):
            # Do not save experimental results
            if self.output_path.split('/')[0] in f:
                continue
            src_file = os.path.join(src, f)
            file_split = f.split(".")
            # Save '.py' file and config file of this experiment
            if (len(file_split) >= 2 and file_split[1] == "py") or f == config_path.split('/')[-1]:
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                dst_file = os.path.join(dst, f)
                try:
                    shutil.copyfile(src=src_file, dst=dst_file)
                except:
                    print("Copy file error! src: {}, dst: {}".format(src_file, dst_file))
            elif os.path.isdir(src_file):
                deeper_dst = os.path.join(dst, f)
                self.save_code(config_path=config_path, src=src_file, dst=deeper_dst)
