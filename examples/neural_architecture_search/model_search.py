from operations import *
import jittor as jit
from jittor import nn
from operations import *  # 假设 operations 模块已经转换为 Jittor 兼容
from genotypes import PRIMITIVES, Genotype  # 假设 genotypes 模块已经转换为 Jittor 兼容
import collections


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = []
        k=0
        for primitive in PRIMITIVES:
            k+=1
            op = OPS[primitive](C, stride, False)
            if "pool" in primitive:
                # op = MySequential('op', op, nn.BatchNorm(C, affine=False))
                op = nn.Sequential(op, nn.BatchNorm(C, affine=False))
            # else:
            #     self.add_module(f"mixed_op_{k}", op) 
            self._ops.append(op)

    def execute(self, x, weights):
        # Initialize a variable to accumulate the sum
        result = 0

        # Loop over the weights and operations in self._ops
        for w, op in zip(weights, self._ops):
            # Compute the product of weight and the output of op(x)
            product = w * op(x)
            # print(type(w))
            # print(type(op(x)))
            # Add the product to the result
            result += product

        # Return the final accumulated result
        return result
        # return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = []
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                # self.add_module(f"op_{i}_{j}", op) 
                self._ops.append(op)

    def execute(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return jit.concat(states[-self._multiplier:], dim=1)


# class MySequential(nn.Module):
#     def __init__(self, *args):
#         super().__init__()
#         # 如果传入的是单个列表或元组，解开并逐一注册
#         if len(args) == 1 and isinstance(args[0], (list, tuple)):
#             args = args[0]
#         self._ordered_modules = []  # 显式维护一个有序列表
#         for idx, module in enumerate(args):
#             self.add_module(str(idx), module)  # 自动注册子模块
#             self._ordered_modules.append(str(idx))  # 记录模块的注册顺序

#     def execute(self, x):
#         for name in self._ordered_modules:  # 遍历有序模块
#             x = self._modules[name](x)
#         return x

#     def __iter__(self):
#         """支持迭代"""
#         return (self._modules[name] for name in self._ordered_modules)

#     def __len__(self):
#         return len(self._ordered_modules)

#     def __getitem__(self, idx):
#         """支持通过索引访问"""
#         if isinstance(idx, int):
#             if idx < 0 or idx >= len(self._ordered_modules):
#                 raise IndexError(f"Index {idx} is out of range")
#             key = self._ordered_modules[idx]
#             return self._modules[key]
#         else:
#             raise TypeError("Index must be an integer")



class MySequential(nn.Module):
    def __init__(self, name='sequential',*args):
        super().__init__()
        self.layers = collections.OrderedDict()  # 显式使用 layers 属性以保持兼容性
        # 如果传入的是单个列表或元组，解开并逐一注册
        # if len(args) == 1 and isinstance(args[0], (list, tuple)):
        #     args = args[0]
        # elif len(args) == 1 and isinstance(args[0], collections.OrderedDict):
        #     for name, module in args[0].items():
        #         self.add_module(name, module)
        # else:
        #     for idx, module in enumerate(args):
        #         self.add_module(str(idx), module)
        
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(name+'_'+str(idx), module)

    def execute(self, x):
        """依次执行子模块"""
        for layer in self.layers.values():
            x = layer(x)
        return x

    # def append(self, module):
    #     """追加模块"""
    #     name = str(len(self.layers))
    #     self.add_module(name, module)

    # # def add_module(self, name, module):
    # #     """新增模块并记录到 layers"""
    # #     if not isinstance(name, str):
    # #         raise TypeError("Module name must be a string")
    # #     if name in self.layers:
    # #         raise KeyError(f"Module {name} already exists")
    # #     self.layers[name] = module
    # #     super().add_module(name, module)

    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), f"Module is not a type"
        self.layers[str(name)]=mod
        super().add_module(str(name), mod)

    # def __getitem__(self, idx):
    #     """通过索引或名称访问子模块"""
    #     if isinstance(idx, int):
    #         if idx < 0 or idx >= len(self.layers):
    #             raise IndexError(f"Index {idx} is out of range")
    #         return list(self.layers.values())[idx]
    #     elif isinstance(idx, str):
    #         if idx not in self.layers:
    #             raise KeyError(f"Module {idx} not found")
    #         return self.layers[idx]
    #     else:
    #         raise TypeError("Index must be an integer or string")

    # def __iter__(self):
    #     """支持迭代"""
    #     return iter(self.layers.values())

    # def __len__(self):
    #     """返回模块个数"""
    #     return len(self.layers)

    # def named_children(self):
    #     """返回子模块名称和模块"""
    #     return list(self.layers.items())

    # def keys(self):
    #     """返回模块名称"""
    #     return self.layers.keys()

    # def values(self):
    #     """返回模块"""
    #     return self.layers.values()

    # def items(self):
    #     """返回模块名称和模块"""
    #     return self.layers.items()

    # def __getattr__(self, key):
    #     """支持通过属性名访问模块"""
    #     if key in self.layers:
    #         return self.layers[key]
    #     return super().__getattr__(key)



class Network(nn.Module):

    def __init__(self, C, num_classes, layers_num, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers_num = layers_num
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        # self.stem = MySequential('stem',
        #     nn.Conv(3, C_curr, 3, padding=1, bias=False),
        #     nn.BatchNorm(C_curr)
        # )
        self.stem = MySequential('stem', nn.Conv(3, C_curr, 3, padding=1, bias=False),nn.BatchNorm(C_curr))

        # print('module_test',self.stem._modules)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # self.cells = nn.ModuleList()
        self.cells = []
        reduction_prev = False
        for i in range(layers_num):
            if i in [layers_num // 3, 2 * layers_num // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            # self.cells.append(cell)
            # self.add_module(f"cell_{i}", cell) 
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        # print('cell_test',self.cells)
        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers_num, self._criterion)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    # def execute(self, input):
    #     s0 = s1 = self.stem(input)
    #     for i, cell in enumerate(self.cells):
    #         if cell.reduction:
    #             weights = jit.nn.softmax(self.alphas_reduce, dim=-1)
    #         else:
    #             weights = jit.nn.softmax(self.alphas_normal, dim=-1)
    #         s0, s1 = s1, cell(s0, s1, weights)
    #     out = self.global_pooling(s1)
    #     logits = self.classifier(out.view(out.size(0), -1))
    #     return logits

    def execute(self, input):
        s0 = s1 = self.stem(input)
        # print(f"After stem: s0 shape: {s0.shape}, s1 shape: {s1.shape}")
        # print(s0.dtype,s1.dtype)

        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = jit.nn.softmax(self.hidden_params['alphas_reduce'], dim=-1)
            else:
                weights = jit.nn.softmax(self.hidden_params['alphas_normal'], dim=-1)
            # print(f"After softmax at cell {i}: weights shape: {weights.shape}")
            # print(s0.dtype,s1.dtype,weights.dtype)
            s0, s1 = s1, cell(s0, s1, weights)
            # print(f"After cell {i}: s0 shape: {s0.shape}, s1 shape: {s1.shape}")
        
        out = self.global_pooling(s1)
        # print(f"After global pooling: out shape: {out.shape}")
        # out = out.view(out.size(0), -1)  # [batch_size, 256]
        # print('Wrong!!!',out.shape[0],out.shape[1])
        # assert out.shape[1] == self.classifier.in_features, (
        #     f"Expected {self.classifier.in_features}, but got {out.shape[1]}"
        # )
        # print('right!!!',self.classifier.in_features)
        # print('right!!!',self.classifier.out_features)
        # print('right!!!',self.classifier.weight.shape)
        # print('right!!!',self.classifier.bias.shape)
        mid_out = out.view(out.size(0), -1)
        # print(mid_out.shape)
        logits = self.classifier(mid_out)
        # print(f"After classifier: logits shape: {logits.shape}")
        
        return logits


    def _loss(self, input, target):
        logits = self.execute(input)
        return self._criterion(logits, target)

    # def _initialize_alphas(self):
    #     k = sum(1 for i in range(self._steps) for n in range(2 + i))
    #     num_ops = len(PRIMITIVES)

    #     self.alphas_normal = jit.array(1e-3 * jit.randn(k, num_ops))
    #     self.alphas_reduce = jit.array(1e-3 * jit.randn(k, num_ops))
    #     self._arch_parameters = [
    #         self.alphas_normal,
    #         self.alphas_reduce,
    #     ]
    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.hidden_params = {
            "alphas_normal": jit.array(1e-3 * jit.randn(k, num_ops)),
            'alphas_reduce':jit.array(1e-3 * jit.randn(k, num_ops))
        }

        self._arch_parameters = list(self.hidden_params.values())
        
    # def _initialize_alphas(self):
    #     k = sum(1 for i in range(self._steps) for n in range(2 + i))
    #     num_ops = len(PRIMITIVES)

    #     # 将变量存储为普通属性，而不是 Jittor 自动管理的参数
    #     self.__dict__["alphas_normal"] = jit.array(1e-3 * jit.randn(k, num_ops))
    #     self.__dict__["alphas_reduce"] = jit.array(1e-3 * jit.randn(k, num_ops))
    #     self._arch_parameters = [
    #         self.__dict__["alphas_normal"],
    #         self.__dict__["alphas_reduce"],
    #     ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(
                    range(i + 2),
                    key=lambda x: -max(
                        W[x][k]
                        for k in range(len(W[x]))
                        if k != PRIMITIVES.index("none")
                    ),
                )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index("none"):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(jit.nn.softmax(self.hidden_params['alphas_normal'], dim=-1).data)
        gene_reduce = _parse(jit.nn.softmax(self.hidden_params['alphas_reduce'], dim=-1).data)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )
        return genotype