from __future__ import division

import os.path as osp
import time

import numpy as np
import torch
from torch import nn
from ..registry import NECKS

import math
import torch.nn.functional as F
from prodict import Prodict

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

if TORCH_VERSION > (1, 4):
    BatchNorm2d = torch.nn.BatchNorm2d
else:

    class BatchNorm2d(torch.nn.BatchNorm2d):
        """
        A wrapper around :class:`torch.nn.BatchNorm2d` to support zero-size tensor.
        """

        def forward(self, x):
            if x.numel() > 0:
                return super(BatchNorm2d, self).forward(x)
            # get output shape
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels)

def get_activation(activation):
    """
    Args:
        norm (str or callable):

    Returns:
        nn.Module or None: the normalization layer
    """
    if activation is None:
        return None

    atype = activation.NAME
    inplace = activation.INPLACE
    act = {
        "ReLU": nn.ReLU,
        "ReLU6": nn.ReLU6,
    }[atype]
    return act(inplace=inplace)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """
    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        if x.numel() == 0 and TORCH_VERSION <= (1, 4):
            assert not isinstance(
                self.norm, torch.nn.GroupNorm
            ), "GroupNorm does not support empty inputs in PyTorch <=1.4!"
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride=1, 
        norm="BN", 
        activation=None
    ):
        super().__init__()

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        self.activation = get_activation(activation)

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

    def forward(self, x, gate):
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out = gate(out, x) + shortcut
        out = self.activation(out)
        del shortcut
        torch.cuda.empty_cache()
        return out

class SpatialGate(nn.Module):
    def __init__(
        self,
        in_channels,
        num_groups=1,
        kernel_size=1,
        padding=0,
        stride=1,
        gate_activation="ReTanH",
        gate_activation_kargs=None,
        get_running_cost=None
    ):
        super(SpatialGate, self).__init__()
        self.num_groups = num_groups
        self.gate_conv = nn.Conv2d(in_channels,
                                   num_groups,
                                   kernel_size,
                                   padding=padding,
                                   stride=stride)
        self.gate_activation = gate_activation
        self.gate_activation_kargs = gate_activation_kargs
        if gate_activation == "ReTanH":
            self.gate_activate = lambda x : torch.tanh(x).clamp(min=0)
        elif gate_activation == "Sigmoid":
            self.gate_activate = lambda x : torch.sigmoid(x)
        elif gate_activation == "GeReTanH":
            assert "tau" in gate_activation_kargs
            tau = gate_activation_kargs["tau"]
            ttau = math.tanh(tau)
            self.gate_activate = lambda x : ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()
        self.get_running_cost = get_running_cost
        self.running_cost = None
        self.init_parameters()

    def init_parameters(self, init_gate=0.99):
        if self.gate_activation == "ReTanH":
            bias_value = 0.5 * math.log((1 + init_gate) /  (1 - init_gate))
        elif self.gate_activation == "Sigmoid":
            bias_value = 0.5 * math.log(init_gate /  (1 - init_gate))
        elif self.gate_activation == "GeReTanH":
            tau = self.gate_activation_kargs["tau"]
            bias_value = 0.5 * math.log((1 + init_gate * math.exp(2 * tau)) /  (1 - init_gate))
        nn.init.normal_(self.gate_conv.weight, std=0.01)
        nn.init.constant_(self.gate_conv.bias, bias_value)

    def encode(self, *inputs):
        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return tuple(outputs)

    def decode(self, *inputs):
        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return tuple(outputs)
    
    def update_running_cost(self, gate):
        if self.get_running_cost is not None:
            cost = self.get_running_cost(gate)
            if self.running_cost is not None:
                self.running_cost = [x + y for x, y in zip(self.running_cost, cost)]
            else:
                self.running_cost = cost
            del cost
            torch.cuda.empty_cache()

    def clear_running_cost(self):
        self.running_cost = None

    def forward(self, data_input, gate_input):
        gate = self.gate_activate(self.gate_conv(gate_input))
        self.update_running_cost(gate)
        data, gate = self.encode(data_input, gate)
        output, = self.decode(data * gate)
        del data, gate
        torch.cuda.empty_cache()
        return output


class DynamicBottleneck(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        kernel_size=1,
        padding=0,
        stride=1,
        num_groups=1,
        norm="GN",
        gate_activation="ReTanH",
        gate_activation_kargs=None 
    ):
        super(DynamicBottleneck, self).__init__()
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = BasicBlock(in_channels,
                                     out_channels,
                                     stride=stride,
                                     norm=norm,
                                     activation=Prodict(NAME="ReLU", INPLACE=True))
        self.gate = SpatialGate(in_channels,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs=gate_activation_kargs,
                                get_running_cost=self.get_running_cost)
        self.init_parameters()

    def init_parameters(self):
        # self.gate.init_parameters()
        for layer in self.bottleneck.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def get_running_cost(self, gate):
        conv_costs = [x * 3 ** 2 for x in [self.in_channels * self.out_channels, self.out_channels ** 2]]
        if self.in_channels != self.out_channels:
            conv_costs[-1] += self.in_channels * out_channels
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_costs = [conv_cost + norm_cost for conv_cost in conv_costs]
        del conv_costs, norm_cost
        running_cost = None
        for unit_cost in unit_costs[::-1]:
            num_groups = gate.shape[1]
            hard_gate = (gate != 0).float()
            cost = [gate * unit_cost / num_groups,
                    hard_gate * unit_cost / num_groups,
                    torch.ones_like(gate) * unit_cost / num_groups]
            cost = [x.flatten(1).sum(-1) for x in cost]
            gate = F.max_pool2d(gate, kernel_size=3, stride=1, padding=1)
            gate = gate.max(dim=1, keepdim=True).values
            if running_cost is None:
                running_cost = cost
            else:
                running_cost = [x + y for x, y in zip(running_cost, cost)]
        del unit_costs, cost, gate
        torch.cuda.empty_cache()
        return tuple(running_cost)

    def forward(self, input):
        output = self.bottleneck(input, self.gate)
        # output = self.gate(data, input)
        return output


class DynamicConv2D(nn.Module):
    def __init__(
        self, 
        in_channels,
        out_channels,
        num_convs,
        kernel_size=1,
        padding=0,
        stride=1,
        num_groups=1,
        norm="GN",
        gate_activation="ReTanH",
        gate_activation_kargs=None,
        depthwise=False
    ):
        super(DynamicConv2D, self).__init__()
        if depthwise:
            assert in_channels == out_channels
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        convs = []
        for _ in range(num_convs):
            convs += [nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=in_channels if depthwise else 1),
                      get_norm(norm, in_channels)]
            in_channels = out_channels
        self.convs = nn.Sequential(*convs)
        self.gate = SpatialGate(in_channels,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs=gate_activation_kargs,
                                get_running_cost=self.get_running_cost)
        self.init_parameters()

    def init_parameters(self):
        self.gate.init_parameters()
        for layer in self.convs.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.GroupNorm):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.bias, 0)

    def get_running_cost(self, gate):
        if self.depthwise:
            conv_cost = self.in_channels * len(self.convs) * \
                self.kernel_size ** 2
        else:
            conv_cost = self.in_channels * self.out_channels * len(self.convs) * \
                self.kernel_size ** 2
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_cost = conv_cost + norm_cost
        del conv_cost, norm_cost
        torch.cuda.empty_cache()
        hard_gate = (gate != 0).float()
        cost = [gate.detach() * unit_cost / self.num_groups,
                hard_gate * unit_cost / self.num_groups,
                torch.ones_like(gate) * unit_cost / self.num_groups]
        del unit_cost
        torch.cuda.empty_cache()
        cost = [x.flatten(1).sum(-1) for x in cost]
        return tuple(cost)

    def forward(self, input):
        data = self.convs(input)
        output = self.gate(data, input)
        del data
        torch.cuda.empty_cache()
        return output


class DynamicScale(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs=1,
        kernel_size=1,
        padding=0,
        stride=1,
        num_groups=1,
        num_adjacent_scales=2,
        depth_module: nn.Module=None,
        resize_method="bilinear",
        norm="GN",
        gate_activation="ReTanH",
        gate_activation_kargs=None
    ):
        super(DynamicScale, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [DynamicConv2D(
            in_channels,
            out_channels,
            num_convs=num_convs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            num_groups=num_groups,
            norm=norm,
            gate_activation=gate_activation,
            gate_activation_kargs=gate_activation_kargs,
            depthwise=True
        ) for _ in range(num_adjacent_scales)]
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        if resize_method == "bilinear":
            self.resize = lambda x, s : F.interpolate(
                x, size=s, mode="bilinear", align_corners=True)
        else:
            raise NotImplementedError()
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))

    def init_weights(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs):
        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x) for m in self.dynamic_convs])
        outputs = []
        for l, x in enumerate(inputs):
            scale_feature = []
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-2:])
                    scale_feature.append(feature)
            scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            if self.depth_module is not None:
                scale_feature = self.depth_module(scale_feature)
            outputs.append(scale_feature)
        del dynamic_scales,feature, scale_feature
        torch.cuda.empty_cache()
        return tuple(outputs)

@NECKS.register_module
class DynamicRoute(nn.Module):

    def __init__(self,
                 in_channels,
                 dynamic_depth,
                 gate_activation='GeReTanH',
                 gate_activation_kargs=dict(tau=1.5),
                 num_groups=1,
                 resize_method='bilinear'):
        super(DynamicRoute, self).__init__()
        self.in_channels = in_channels
        subnet = []
        for _ in range(dynamic_depth):
            subnet_conv = DynamicBottleneck(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm="GN",
                num_groups=num_groups,
                gate_activation=gate_activation,
                gate_activation_kargs=gate_activation_kargs
            )
            subnet.append(
                DynamicScale(
                    in_channels,
                    in_channels,
                    num_convs=1,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    num_groups=num_groups,
                    num_adjacent_scales=2,
                    resize_method=resize_method,
                    depth_module=subnet_conv,
                    gate_activation=gate_activation,
                    gate_activation_kargs=gate_activation_kargs
                )
            )
        self.subnet = nn.Sequential(*subnet)

    def init_weights(self):
        for layer in self.subnet:
            layer.init_weights()

    def forward(self, feats):
        result = self.subnet(feats)
        return tuple(result)