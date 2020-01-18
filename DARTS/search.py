# @Time  :2020/1/13
# @Author:LiuZhQ

import time
import pyhocon
import argparse


def init():
    parser = argparse.ArgumentParser('DARTS')
    parser.add_argument('--config', '-c', type=str, default='./config/search/template.hocon',
                        help='Path of Config file.')
    parser.add_argument('--output', '-o', type=str, default='result', help='Path of output result file.')
    args = parser.parse_args()
    _date = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
    result_path = args.output + '_' + _date
    hocon = pyhocon.ConfigFactory.parse_file(args.config)
    hocon.result_path = result_path
    return hocon


def main():
    pass


if __name__ == '__main__':
    hocon = init()
    print(hocon.result_path)
