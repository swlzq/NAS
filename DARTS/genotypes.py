# @Author: LiuZhQ

from collections import namedtuple

# operation的名称
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',  # separable convolution
    'sep_conv_5x5',
    'dil_conv_3x3',  # dilated separable convolution
    'dil_conv_5x5'
]


# 定义一个Genotype元组，concat？？？
Genotype = namedtuple('Genotype', ['normal', 'normal_concat', 'reduce', 'reduce_concat'])


