# @Author: LiuZhQ

import torch.nn.functional as F
from model.operations import *

from model.genotypes import PRIMITIVES
from model.genotypes import Genotype


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C  # 通道数
        self._num_classes = num_classes  # 类别数
        self._layers = layers  # cell的数量
        self._steps = steps  # 用来搜索的节点数量
        # 用于通道数增加的倍数（这里，每个cell的输出等于后四个中间节点的输出concatenate，
        # 因此，通道数=4*单个node的通道数）
        self._multiplier = multiplier
        self._criterion = criterion

        # stem层固定，不通过搜索获得,是第一个cell的前两个节点
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # 通道数更新,C_prev_prev是第一个预处理的通道数（preprocess0）,
        # C_prev是第二个预处理的通道数（preprocess1）, C_curr是operations的通道数
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False  # 前一个是否reduction模块
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:  # 1/3和2/3处为reduction模块
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            # 通道赋值，multiplier*C_curr即为最终输出的通道数
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)  # 为什么是C_prev???

        self._initialize_alphas()  # 随机初始化alphas参数

    def forward(self, x):
        s0 = s1 = self.stem(x)  # 预定义的前两个node
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            # 第k个cell的前两个节点的输入是第k-2和第k-1的输出
            s0, s1 = s1, cell(s0, s1, weights)
        x = self.global_pooling(s1)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        # 随机初始化14*7的数组,乘以1e-3,结果趋近于0,套上softmax之后概率就约等于1/num_ops
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # 解析权重
        def _parse(weights):
            gene = []
            n = 2  # 预定义两个输入节点
            start = 0  # 当前第start个中间节点
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                # 对边进行排序，选择每条边最大的权重进行排序，-max降序排列，取权重靠前的两条边
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                                                if k != PRIMITIVES.index('none')))[:2]
                # 选取权重最大的操作
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end  # 下一个中间节点
                n += 1  # 可选择的边加1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        # concat在一起的node的序号
        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        # preprocess重复两次，Nas-Net调出来的，affine为什么是False？
        # FactorizedReduce有什么用
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()  # 这有什么用？后面没出现过
        for i in range(self._steps):  # 按照node循环添加edge
            for j in range(2 + i):
                # Cells located at the 1/3 and 2/3 of the total depth of the network are reduction cells,
                # in which all the operations adjacent to the input nodes are of stride two.
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        # 输入前两个节点，先做预处理
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]  # 存放已计算好的节点
        offset = 0  # 当前操作在列表中的位置偏移

        # 按顺序计算每个节点的输出
        # 每个节点的输出等于前面节点输出之和
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)  # 计算好该节点，存入states，开始计算下一节点

        return torch.cat(states[-self._multiplier:], dim=1)  # 后四个节点cat作为最终输出


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # 遍历所有的operation
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        # 混合操作的输出为：sum(操作的权重*操作的值)
        return sum(w * op(x) for w, op in zip(weights, self._ops))
