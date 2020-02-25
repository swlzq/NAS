# @Author: LiuZhQ

import argparse

from setting import EvaluateSetting
from trainer import CifarTrainer
from utils import set_randomness

if __name__ == '__main__':
    # Load config path
    parser = argparse.ArgumentParser(description='Neural Architecture Search')
    parser.add_argument('--config_path', type=str, metavar='config_path',
                        default='config/evaluate/base.hocon',
                        help='The path of config file for training.')
    args = parser.parse_args()
    config_path = args.config_path
    # Load experimental setting
    setting = EvaluateSetting(config_path)
    for i in range(setting.repeat):
        setting.set_output_path(i)
        logger = setting.get_logger(logger_name='search{}'.format(i))
        setting.save_setting()
        setting.save_code(config_path=config_path)
        set_randomness(setting.seed + i)
        CifarTrainer(setting, logger).run()
